import numpy as np
import math
import random

from multiagent_rlrm.learning_algorithms.learning_algorithm import BaseLearningAlgorithm


class RMax(BaseLearningAlgorithm):
    def __init__(
        self,
        gamma=0.99,
        s_a_threshold=100,
        max_reward=1.0,
        epsilon_one=0.99,
        VI_delta=1e-4,
        VI_delta_rel=False,  # wheather delta in VI is relative to current values
        use_qrm=False,  # Use qrm experience
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.s_a_threshold = s_a_threshold
        self.max_reward = max_reward
        self.gamma = gamma
        self.epsilon_one = epsilon_one
        self.use_qrm = use_qrm

        # value iteration parameters
        self.delta_value_iter = VI_delta
        self.delta_rel = VI_delta_rel
        self.max_num_value_iter = 1e4
        # math.ceil(
        #    np.log(1 / (self.epsilon_one * (1 - self.gamma))) / (1 - self.gamma)
        # )
        self.requires_rm_event = False

        # Compute optimistic value based on max reward and discount factor
        optimistic_value = self.max_reward / (1 - self.gamma)

        # Initialize Q-table with optimistic value
        self.q_table = np.full(
            (self.state_space_size, self.action_space_size), optimistic_value
        )

        # self.q_table = np.ones((state_space_size, action_space_size)) * self.max_reward * 1 / (1 - self.gamma)
        self.rewards = np.zeros((self.state_space_size, self.action_space_size))
        self.transitions = np.zeros(
            (self.state_space_size, self.action_space_size, self.state_space_size)
        )
        self.s_a_counts = np.zeros((self.state_space_size, self.action_space_size))

        self.prev_state = None
        self.prev_action = None

        # print(self.q_table, "q_table")
        # breakpoint()

        self.param_str = f"{self.s_a_threshold},{self.gamma:.2f},{self.delta_value_iter},{self.delta_rel}"

        print(f"!!! RMax ({self.param_str}) !!!")

        self.tosave += ["q_table", "rewards", "transitions", "s_a_counts"]
        self.noupdate_cnt = 0

    """def choose_action(self, encoded_state, epsilon=None, **kwargs):
        if random.random() < epsilon:
            # Esplorazione: scegli un'azione casuale
            return random.randint(0, self.action_space_size - 1)
        else:
            # Mescola le azioni per garantire una selezione equa in caso di parità di valori Q
            actions = list(range(self.action_space_size))
            random.shuffle(actions)

            # Inizializza il miglior valore Q e l'azione corrispondente
            max_q_val = float("-inf")
            best_action = None

            # Cerca l'azione con il valore Q massimo
            for action in actions:
                q_val = self.q_table[encoded_state, action]
                if q_val > max_q_val:
                    max_q_val = q_val
                    best_action = action

            # Assicurati che sia stata trovata un'azione migliore
            if best_action is None:
                best_action = random.choice(actions)

        return best_action"""

    """def choose_action(self, encoded_state, epsilon, **kwargs):
        # Controllo esplorazione vs. exploitation con un approccio ε-greedy
        if random.random() < epsilon:
            # Esplorazione: scegli un'azione casuale
            return random.choice(range(self.action_space_size))
        else:
            # Ottieni tutti i valori Q per lo stato corrente
            q_values = self.q_table[encoded_state]

            # Trova il valore Q massimo
            max_q_val = np.max(q_values)

            # Trova tutte le azioni che hanno il valore Q massimo
            actions_with_max_q = np.where(q_values == max_q_val)[0]

            # Scegli un'azione casualmente tra quelle con il valore Q massimo
            return np.random.choice(actions_with_max_q)"""

    def choose_action(self, encoded_state, best=False, rng=None, **kwargs):
        """
        Args:
            state (State)

        Returns:
            (tuple) --> (float, str): where the float is the Qval, str is the action.
        """
        if rng is None:
            rng = self.rng

        # Grab random initial action in case all equal
        actions = list(range(self.action_space_size))
        # random.shuffle(actions)
        if best == False and np.all(
            self.q_table[encoded_state] == self.q_table[encoded_state, 0]
        ):
            best_action = rng.choice(actions)
        else:
            best_action = np.argmax(self.q_table[encoded_state])
        return best_action

    """def choose_action(self, encoded_state, epsilon=0.1):
        if random.random() < 0.1:
            # Esplorazione: scegli un'azione casuale
            return random.choice(range(self.action_space_size))
        else:
            actions = list(range(self.action_space_size))
            random.shuffle(actions)
            # Controlla se tutti i valori Q per lo stato corrente sono uguali
            if np.all(self.q_table[encoded_state] == self.q_table[encoded_state][0]):
                # Se tutti i valori Q sono uguali, scegli un'azione a caso per incoraggiare l'esplorazioneactions
                return random.choice(actions)
            else:
                # Altrimenti, scegli l'azione con il valore Q massimo
                return np.argmax(self.q_table[encoded_state])"""

    def update(self, state, next_state, action, reward, terminated, **kwargs):
        if state == None or action == None:
            return False

        terminate = False
        self.noupdate_cnt += 1
        if self.noupdate_cnt % 10000 == 0:
            print(f"no update count: {self.noupdate_cnt}")
            if self.noupdate_cnt >= 1e6:
                terminate = True

        def update_q(state, action, reward, next_state, terminated):
            if terminated:
                self.q_table[next_state, :] = 0
                self.transitions[next_state, :, :] = 0
                self.transitions[next_state, :, next_state] = self.s_a_threshold
            if self.s_a_counts[state][action] < self.s_a_threshold:
                self.rewards[state][action] += reward
                self.s_a_counts[state][action] += 1
                self.transitions[state][action][next_state] += 1
                if self.s_a_counts[state][action] == self.s_a_threshold:
                    self.value_iteration()
                    self.noupdate_cnt = 0
                    self.count_known = self.s_a_counts >= self.s_a_threshold
                    # print(f"Known: {np.sum(self.count_known)}/{self.state_space_size*self.action_space_size}")

        # Handle counterfactual experiences if QRM is enabled

        update_q(state, action, reward, next_state, terminated)

        if self.use_qrm:
            qrm_experiences = kwargs.get("info", {}).get("qrm_experience", [])
            # print(len(qrm_experiences))
            for qrm_exp in qrm_experiences:
                _s, _a, _r, _sn, _done, _, _, _, _, _ = qrm_exp
                update_q(_s, _a, _r, _sn, _done)

        return terminate

    def value_iteration(self):
        """
        Do some iterations of value iteration to compute the q values
        Only update the (s, a) pairs that have enough experiences seen
        Q(s, a) = R(s, a) + gamma * \sum_s' T(s, a, s') * max_a' Q(s', a')
        """
        # mask for update
        mask = self.s_a_counts >= self.s_a_threshold
        pseudo_count = np.where(self.s_a_counts == 0, 1, self.s_a_counts)

        # build the reward model
        empirical_reward_mat = self.rewards / pseudo_count

        # build the transition model: assume self-loop if there's not enough data
        # assume a self-loop if there's not enough data
        empirical_transition_mat = self.transitions / pseudo_count[:, :, None]
        # only masked positions should be trusted, otherwise self transition

        empirical_transition_mat[~mask] = self._self_transition_mat()[~mask]
        # breakpoint()
        # assert np.allclose(empirical_transition_mat.sum(axis=-1), np.ones_like(empirical_transition_mat.sum(axis=-1)), atol=1e-8)
        assert np.all(np.abs(empirical_transition_mat.sum(axis=-1) - 1) < 1e-6)

        for _ in range(int(self.max_num_value_iter)):
            v = np.max(self.q_table, axis=-1)
            new_q = empirical_reward_mat + self.gamma * np.einsum(
                "san,n->sa", empirical_transition_mat, v
            )
            # absolute/relative error (for env with small Q values)
            # possible RuntimeWarning: invalid value encountered in divide
            qabsdiff = np.abs(self.q_table[mask] - new_q[mask])

            if self.delta_rel:  # check convergence
                m = self.q_table[mask] != 0
                qreldiff = np.zeros_like(qabsdiff)
                np.place(qreldiff, m, qabsdiff[m] / np.abs(self.q_table[mask][m]))
                br = np.all(qreldiff < self.delta_value_iter)
            else:
                br = np.all(qabsdiff < self.delta_value_iter)
            if br:
                break

            self.q_table[mask] = new_q[mask]  # Aggiorna solo dove mask è True.

    def _self_transition_mat(self):
        """
        Create a transition matrix where each state transitions to itself with probability 1.
        This matrix is used for state-action pairs that haven't been visited enough times.
        """
        # Initialize a transition matrix of zeros with the same shape as `self.transitions`.
        self_transition_mat = np.zeros_like(self.transitions)
        self_transition_mat[
            np.arange(self.state_space_size), :, np.arange(self.state_space_size)
        ] = 1
        return self_transition_mat

    def get_q_value(self, state, action):
        return self.q_table[state][action]
