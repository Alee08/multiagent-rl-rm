from abc import ABC, abstractmethod

import numpy as np


class BaseLearningAlgorithm(ABC):
    """Minimal interface shared by the tabular RL algorithms in this repo."""

    def __init__(
        self, state_space_size, action_space_size, gamma=0.99, seed=2020, max_steps=400
    ):
        """
        Initialize shared learning parameters.

        Args:
            state_space_size (int): Number of encoded environment states.
            action_space_size (int): Number of discrete actions.
            gamma (float): Discount factor.
            seed (int): RNG seed.
            max_steps (int): Maximum steps per episode during learning (used only if a loop adds it).
        """
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.episode = 0
        self.max_steps = max_steps  # left for algorithms that check a cap
        self.verbose = 0
        self.seed = seed
        self.gamma = gamma
        self.rleval = None
        self.user_quit = False
        self.agent_quit = False
        self.rng = np.random.default_rng(seed=self.seed)
        # Subclasses extend this for ad-hoc save/load helpers.
        self.tosave = ["rng"]

    @abstractmethod
    def update(
        self, encoded_state, encoded_next_state, action, reward, terminated, **kwargs
    ):
        """
        Update the learning algorithm with a single transition.

        Args:
            encoded_state (int or np.ndarray): Current encoded observation.
            encoded_next_state (int or np.ndarray): Next encoded observation.
            action (int): Action index taken.
            reward (float): Observed reward (can include shaping).
            terminated (bool): True if the environment signaled termination (not truncation).
            **kwargs: Extra info (e.g., info dict from env).
        """
        pass

    @abstractmethod
    def choose_action(self, encoded_state, **kwargs):
        """
        Select an action given the encoded state.

        Args:
            encoded_state (int or np.ndarray): Encoded observation.
            **kwargs: Optional extra info for exploration policies.

        Returns:
            int: Chosen action index.
        """
        pass

    def learn_init(self):
        """Optional hook executed once before the learning loop begins."""
        pass

    def learn_init_episode(self):
        """Optional hook executed at the start of each episode."""
        pass

    # def learn_update(self, obs, action: int, reward: float, terminated: bool, next_obs, info):
    #    pass

    def learn_done_episode(self):
        """Optional hook executed at the end of each episode."""
        pass

    def learn_end(self):
        """Optional hook executed after learning is finished."""
        pass
