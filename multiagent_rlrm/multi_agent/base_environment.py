from pettingzoo import ParallelEnv
from gymnasium.spaces import Discrete, MultiDiscrete
from unified_planning.shortcuts import *
from unified_planning.model.multi_agent import *
import numpy as np


class BaseEnvironment(ParallelEnv, MultiAgentProblem):
    """
    A minimal base environment for multi-agent RL, integrating PettingZoo's ParallelEnv
    and Unified-Planning's MultiAgentProblem.

    Child classes should:
      - Populate `self.agents` appropriately.
      - Override `reset(...)` and `step(...)` for domain-specific behavior.
      - Optionally override `apply_actions(...)`, `_calculate_rewards()`,
        `check_terminations(...)`, and `_generate_observations()` to incorporate domain logic.

    Attributes:
      - grid_width (int): The width of the grid or environment (if applicable).
      - grid_height (int): The height of the grid or environment (if applicable).
      - agents (list): List of agents (e.g., `AgentRL` instances).
      - active_agents (dict): Maps agent.name -> bool (True if active).
      - agent_fail (dict): Maps agent.name -> bool (True if agent has failed).
      - agent_steps (dict): Maps agent.name -> integer step count per agent.
      - timestep (int): A global timestep counter.
    """

    def __init__(self, width: int, height: int):
        """
        Initializes the base environment with a given width and height.
        Child classes may store additional domain parameters here.
        """
        # Initialize parents
        ParallelEnv.__init__(self)
        MultiAgentProblem.__init__(self)

        # Basic geometry (used by some child classes)
        self.grid_width = width
        self.grid_height = height

        # Initialize containers for multi-agent management
        self.active_agents = {}
        self.agent_fail = {}
        self.agent_steps = {}

        self.timestep = 0

    def reset(self, seed=None, options=None):
        """
        Resets the environment to start a new episode.

        Typically:
          1) Possibly reseed the environment.
          2) Reset each agent's internal state (e.g., call agent.reset()).
          3) Set self.timestep = 0, and mark each agent as active (agent_fail[...] = False).
          4) Return (observations, infos).
        """
        self.timestep = 0
        for ag in self.agents:
            self.active_agents[ag.name] = True
            self.agent_fail[ag.name] = False
            self.agent_steps[ag.name] = 0
            # Optionally call ag.reset() if the agent needs its own reset logic.

        # Return observations and infos
        observations = self._generate_observations()
        infos = self._generate_infos()
        return observations, infos

    def step(self, actions):
        """
        Executes one environment step with the provided actions for each agent.

        Typically:
          1) Apply domain logic in apply_actions(actions).
          2) Compute rewards in _calculate_rewards().
          3) Check for terminations in check_terminations().
          4) Generate observations and infos.

        Returns:
          observations: dict of agent.name -> observation
          rewards: dict of agent.name -> float
          terminations: dict of agent.name -> bool
          truncations: dict of agent.name -> bool
          infos: dict of agent.name -> any extra info
        """
        # 1) Apply domain logic
        self.apply_actions(actions)

        # 2) Compute rewards
        rewards = self._calculate_rewards()

        # 3) Check terminations/truncations
        terminations, truncations = self.check_terminations()

        # 4) Generate new observations and infos
        observations = self._generate_observations()
        infos = self._generate_infos()

        self.timestep += 1
        return observations, rewards, terminations, truncations, infos

    def apply_actions(self, actions):
        """
        A hook to apply domain-specific effects of chosen actions to the environment state.
        By default, does nothing. Child classes should override this to implement
        movement, collisions, or other logic.

        Args:
          actions (dict): a mapping agent.name -> action object
        """
        # Example pseudo-code (child class):
        # for ag in self.agents:
        #     if self.active_agents[ag.name]:
        #         action_chosen = actions[ag.name]
        #         ... apply logic ...
        pass

    def _calculate_rewards(self):
        """
        Computes rewards for each agent. By default, returns zero for all agents.
        Child classes should override with domain-based logic.

        Returns:
          dict: agent.name -> float
        """
        return {ag.name: 0.0 for ag in self.agents}

    def check_terminations(self):
        """
        Determines whether each agent is done or truncated.
        By default, sets termination if agent_fail[ag.name] == True,
        and leaves truncation as False. Child classes can override or expand
        (e.g., max steps, final states, etc.).

        Returns:
          (terminations, truncations) where each is a dict agent.name -> bool
        """
        terminations = {}
        truncations = {}
        for ag in self.agents:
            done = self.agent_fail.get(ag.name, False)
            terminations[ag.name] = done
            truncations[ag.name] = False
        return terminations, truncations

    def _generate_observations(self):
        """
        Builds and returns observations for each agent. By default, returns empty dicts.
        Override to reflect current environment state.

        Returns:
          dict: agent.name -> observation
        """
        obs = {}
        for ag in self.agents:
            obs[ag.name] = {}  # e.g., a dictionary containing relevant state
        return obs

    def _generate_infos(self):
        """
        Builds and returns any extra debug info for each agent.
        By default, returns empty dicts.

        Returns:
          dict: agent.name -> info
        """
        infos = {}
        for ag in self.agents:
            infos[ag.name] = {}
        return infos
