import numpy as np
from multiagent_rlrm.learning_algorithms.learning_algorithm import BaseLearningAlgorithm


class QLearningLambda(BaseLearningAlgorithm):
    def __init__(
        self,
        gamma,
        lambd,
        action_selection,
        learning_rate=None,
        epsilon_start=1.0,
        epsilon_end=0.2,
        epsilon_decay=0.99,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lambd = lambd  # Lambda parameter for eligibility traces
        self.action_selection = action_selection  # 'softmax' or 'greedy'
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = self.epsilon_start
        self.q_table = np.zeros((self.state_space_size, self.action_space_size))
        self.e_table = np.zeros(
            (self.state_space_size, self.action_space_size)
        )  # Eligibility trace table
        self.visits = np.zeros((self.state_space_size, self.action_space_size))
        self.tosave += ["q_table", "visits", "epsilon"]

    def update(
        self,
        encoded_state,
        encoded_next_state,
        action,
        reward,
        terminated,
        next_action=None,
        **kwargs,
    ):
        # Visit count used for adaptive step size when learning_rate is None
        self.visits[encoded_state, action] += 1
        lr = (
            self.learning_rate
            if self.learning_rate is not None
            else 1 / self.visits[encoded_state, action]
        )

        # TD error for off-policy Q-learning (uses max over next state)
        best_future_q = (
            0.0 if terminated else float(np.max(self.q_table[encoded_next_state]))
        )
        td_error = (
            reward + self.gamma * best_future_q - self.q_table[encoded_state, action]
        )

        # Replacing traces: set the trace of (s,a) to 1
        self.e_table[encoded_state, action] = 1.0

        # Q update using current traces
        self.q_table += lr * td_error * self.e_table

        # Trace handling following Watkins Q(λ)
        if terminated:
            # Episode ended: wipe traces
            self.e_table.fill(0.0)
        else:
            # Check if the next action is greedy in s'
            if next_action is None:
                next_action = int(np.argmax(self.q_table[encoded_next_state]))
            greedy_next_actions = np.flatnonzero(
                self.q_table[encoded_next_state]
                == np.max(self.q_table[encoded_next_state])
            )
            next_is_greedy = next_action in greedy_next_actions

            if next_is_greedy:
                # Next action is greedy: decay traces
                self.e_table *= self.gamma * self.lambd
            else:
                # Next action is not greedy: wipe traces
                self.e_table.fill(0.0)

    def choose_action(self, encoded_state, best=False, rng=None, **kwargs):
        if best:
            return np.argmax(self.q_table[encoded_state])
        if rng is None:
            rng = self.rng
        if rng.uniform(0, 1) < self.epsilon:
            # Random exploration
            return rng.choice(range(self.action_space_size))
        else:
            # Choose an action based on the configured method
            if self.action_selection == "softmax":
                return self.choose_action_softmax(encoded_state, rng)
            elif self.action_selection == "greedy":
                return self.choose_action_greedy(encoded_state, rng)
            else:
                raise ValueError("Unsupported action selection method")

    def choose_action_softmax(self, encoded_state, rng):
        softmax_probs = self.softmax(self.q_table[encoded_state])
        return rng.choice(np.arange(self.action_space_size), p=softmax_probs)

    def choose_action_greedy(self, encoded_state, rng):
        # choose random action among best ones
        Qa = self.q_table[encoded_state]
        va = np.argmax(Qa)
        maxs = [i for i, v in enumerate(Qa) if v == Qa[va]]
        action = rng.choice(maxs)
        return action  # np.argmax(self.q_table[encoded_state])

    def softmax(self, q_values, beta=4):
        exp_q = np.exp(
            beta * q_values - np.max(beta * q_values)
        )  # Subtract max for numerical stability
        probabilities = exp_q / np.sum(exp_q)
        return probabilities

    def learn_init_episode(self):
        """Reset traces at the beginning of each episode."""
        self.reset_e_table()

    def learn_done_episode(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def reset_e_table(self):
        """Reset eligibility traces after each episode."""
        self.e_table.fill(0)
