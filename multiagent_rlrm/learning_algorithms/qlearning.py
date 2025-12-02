import numpy as np
import random
from multiagent_rlrm.learning_algorithms.learning_algorithm import BaseLearningAlgorithm


class QLearning(BaseLearningAlgorithm):
    """Tabular Q-Learning with optional QRM counterfactuals and reward shaping."""

    def __init__(
        self,
        gamma,
        action_selection,
        learning_rate=None,
        epsilon_start=1.0,
        epsilon_end=0.2,
        epsilon_decay=0.99,
        qtable_init=1,
        use_qrm=False,  # Use qrm experience
        use_rsh=False,  # Use reward shaping
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.action_selection = action_selection  # 'softmax' or 'greedy'
        self.q_table = np.full(
            (self.state_space_size, self.action_space_size), qtable_init, dtype=float
        )
        self.visits = np.zeros((self.state_space_size, self.action_space_size))
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = self.epsilon_start
        self.use_qrm = use_qrm
        self.use_rsh = use_rsh

        self.param_str = f"{learning_rate:0.2f},{gamma:0.2f}"

        self.tosave += ["q_table", "visits", "epsilon"]

    def update(
        self, encoded_state, encoded_next_state, action, reward, terminated, **kwargs
    ):
        """
        Update the Q-value for a transition (s, a, s').

        If reward shaping is enabled and a reward_machine with potentials is provided,
        the shaped reward is computed as:
          R' = R + gamma * Phi(s') - Phi(s)
        """
        rm = kwargs["info"].get("reward_machine", None)
        if (
            self.use_rsh
            and rm is not None
            and hasattr(rm, "potentials")
            and rm.potentials is not None
        ):
            # Assumiamo che nel dizionario info i campi "prev_q" e "q" contengano direttamente
            # le chiavi degli stati della Reward Machine (es. "state0", "state1", …)
            rm_prev_state = kwargs["info"].get("prev_q", None)
            rm_state = kwargs["info"].get("q", None)
            if rm_prev_state is not None and rm_state is not None:
                shaping_reward = self.gamma * rm.potentials.get(
                    rm_state, 0
                ) - rm.potentials.get(rm_prev_state, 0)
                reward += shaping_reward
        # print((f"Update Q-Table: S={encoded_state}, A={action}, R={reward}, S'={encoded_next_state}, Q={new_q}"))

        # Handle counterfactual experiences if QRM is enabled
        def update_q(s, a, r, sn, terminated):
            current_q = self.q_table[s, a]
            self.visits[s, a] += 1
            if self.learning_rate is None:
                lr = 1 / self.visits[s, a]
            else:
                lr = self.learning_rate
            max_future_q = (not terminated) * np.max(self.q_table[sn])
            new_q = (1 - lr) * current_q + lr * (r + self.gamma * max_future_q)
            self.q_table[s, a] = new_q

        # Handle counterfactual experiences if QRM is enabled
        if self.use_qrm:
            qrm_experiences = kwargs.get("info", {}).get("qrm_experience", [])
            for qrm_exp in qrm_experiences:
                #  _s      -> encoded state (from info "prev_q")
                #  _a      -> action index
                #  _r      -> reward (environmental + any shaping already included)
                #  _sn     -> encoded next state
                #  _done   -> termination flag
                #  _current_q -> encoded RM state "prev" (from tuple)
                #  _next_q    -> encoded RM state "next" (from tuple)
                _s, _a, _r, _sn, _done, _, _current_q, _, _next_q, _ = qrm_exp
                if (
                    self.use_rsh
                    and rm is not None
                    and hasattr(rm, "potentials")
                    and rm.potentials is not None
                ):
                    # _current_q is the previous RM state and _next_q is the next RM state
                    rm_prev_state_cf = rm.get_state_from_index(_current_q)
                    rm_state_cf = rm.get_state_from_index(_next_q)
                    shaping_reward_cf = self.gamma * rm.potentials.get(
                        rm_state_cf, 0
                    ) - rm.potentials.get(rm_prev_state_cf, 0)
                    _r += shaping_reward_cf
                update_q(_s, _a, _r, _sn, _done)
        else:
            update_q(encoded_state, action, reward, encoded_next_state, terminated)

        return False  # do not terminate

    def choose_action(self, encoded_state, best=False, rng=None, **kwargs):
        """
        Select an action index using epsilon-greedy or deterministic greedy if best=True.
        """
        if best:
            return np.argmax(self.q_table[encoded_state])
        if rng is None:
            rng = self.rng
        if rng.uniform(0, 1) < self.epsilon:
            # Random exploration
            return rng.choice(range(self.action_space_size))
        else:
            # Choose action according to the configured strategy
            if self.action_selection == "softmax":
                return self.choose_action_softmax(encoded_state, rng)
            elif self.action_selection == "greedy":
                return self.choose_action_greedy(encoded_state, rng)
            else:
                raise ValueError("Unsupported action selection method")

    def choose_action_softmax(self, encoded_state, rng):
        """Sample an action using softmax probabilities over Q-values."""
        softmax_probs = self.softmax(self.q_table[encoded_state])
        return rng.choice(np.arange(self.action_space_size), p=softmax_probs)

    def choose_action_greedy(self, encoded_state, rng):
        """Choose uniformly among the greedy actions."""
        Qa = self.q_table[encoded_state]
        va = np.argmax(Qa)
        maxs = [i for i, v in enumerate(Qa) if v == Qa[va]]
        action = rng.choice(maxs)
        return action  # np.argmax(self.q_table[encoded_state])

    def softmax(self, q_values, beta=4):
        """Compute softmax probabilities with temperature beta."""
        exp_q = np.exp(
            beta * q_values - np.max(beta * q_values)
        )  # Subtract max for numerical stability
        probabilities = exp_q / np.sum(exp_q)
        return probabilities

    def learn_done_episode(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        # print(f"{self.epsilon:6.3f}")

    def reset_epsilon(self):
        """Reset epsilon to its starting value."""
        self.epsilon = self.epsilon_start
