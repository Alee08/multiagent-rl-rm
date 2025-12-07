import numpy as np
import random
from multiagent_rlrm.learning_algorithms.qlearning_lambda import QLearningLambda
from multiagent_rlrm.learning_algorithms.qlearning import QLearning
from multiagent_rlrm.multi_agent.base_environment import BaseEnvironment

random.seed(a=123)
np.random.seed(123)


class MultiAgentFrozenLake(BaseEnvironment):
    metadata = {"name": "multi_agent_frozen_lake"}

    def __init__(self, width, height, holes):
        """
        Multi-agent Frozen Lake environment.

        :param width: Grid width.
        :param height: Grid height.
        :param holes: List of grid positions representing holes.
        """
        super().__init__(width, height)
        self.holes = holes  # Hole positions on the grid
        self.possible_actions = ["up", "down", "left", "right"]
        self.rewards = 0
        self.frozen_lake_stochastic = False  # Whether actions are stochastic
        self.penalty_amount = 0  # Penalty when falling into a hole
        self.active_agents = {agent.name: True for agent in self.agents}
        self.agent_fail = {agent.name: False for agent in self.agents}
        self.agent_steps = {agent.name: 0 for agent in self.agents}
        self.delay_action = False  # Enables delayed (wait) stochastic transitions
        self.epsilon = None
        self.random_start_positions = (
            False  # if True, agents start in random free cells
        )
        # Dedicated RNG for reproducibility across reset/step
        self.rng = np.random.default_rng()

    def reset(self, seed=123, options=None):
        """
        Reset the environment to its initial state.

        Resets:
        - Rewards for each agent
        - Agent active/fail states
        - Agent step counters
        - Each agent’s position and internal state (e.g., Reward Machine)
        """
        self.rewards = {agent.name: 0 for agent in self.agents}
        self.timestep = 0
        self.active_agents = {agent.name: True for agent in self.agents}
        self.agent_fail = {agent.name: False for agent in self.agents}
        self.agent_steps = {agent.name: 0 for agent in self.agents}

        self.rng = (
            np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        )
        rng = self.rng
        if self.random_start_positions:
            start_positions = self._sample_start_positions(rng)
        else:
            # Use the explicitly configured initial positions when not randomizing
            start_positions = [
                getattr(ag, "initial_position", ag.position) for ag in self.agents
            ]

        for agent, start_pos in zip(self.agents, start_positions):
            agent.set_initial_position(*start_pos)
            agent.reset()  # Reset internal RM and states

            l_algo = agent.get_learning_algorithm()

            # For Q(λ), clear the eligibility traces
            if isinstance(l_algo, QLearningLambda):
                l_algo.reset_e_table()

            # Initial state returned by agent
            agent.get_state()

            # Signal end-of-episode for plain Q-learning (if needed)
            if isinstance(l_algo, QLearning):
                l_algo.learn_done_episode()

        # Dummy info dict for initialization
        infos = {agent: {} for agent in self.agents}
        observations = {agent.name: agent.state for agent in self.agents}

        return observations, infos

    def step(self, actions):
        """
        Executes one environment step for all agents.

        :param actions: Dict mapping agent names to chosen actions.
        :return: (observations, rewards, terminations, truncations, infos)
        """
        self.rewards = {a.name: 0 for a in self.agents}
        infos = {a.name: {} for a in self.agents}

        for agent in self.agents:
            rm = getattr(agent, "reward_machine", None)
            rm_done = rm and rm.get_current_state() == rm.get_final_state()

            if not self.active_agents[agent.name] or rm_done:
                # Provide fallback info for inactive agents
                infos[agent.name]["prev_s"] = self.get_state(agent)
                infos[agent.name]["s"] = self.get_state(agent)
                infos[agent.name]["Renv"] = 0
                continue

            current_state = self.get_state(agent)
            ag_action = actions[agent.name]  # Chosen action object

            # Apply the action (deterministic or stochastic)
            if self.frozen_lake_stochastic:
                chosen_name = self.get_stochastic_action(agent, ag_action.name)
                if chosen_name != "wait":
                    self.apply_action(agent, chosen_name)
            else:
                self.apply_action(agent, ag_action.name)

            new_state = self.get_state(agent)

            # Environmental reward from holes
            reward_env = self.holes_in_the_ice(new_state, agent.name)
            reward = reward_env

            self.rewards[agent.name] += reward
            self.agent_steps[agent.name] += 1

            # Log transition info
            infos[agent.name]["prev_s"] = current_state
            infos[agent.name]["s"] = new_state
            infos[agent.name]["Renv"] = reward_env

        self.timestep += 1

        # Determine terminations and truncations
        terminations, truncations = self.check_terminations()

        # Mark inactive agents
        for agent_name in terminations:
            if terminations[agent_name]:
                self.active_agents[agent_name] = False

        observations = {agent.name: agent.state for agent in self.agents}

        return observations, self.rewards, terminations, truncations, infos

    def _sample_start_positions(self, rng):
        """
        Sample distinct start positions not in holes.

        :param rng: np.random.Generator instance.
        :return: List of (x, y) tuples, one per agent.
        """
        free_cells = [
            (x, y)
            for x in range(self.grid_width)
            for y in range(self.grid_height)
            if (x, y) not in self.holes
        ]
        if len(free_cells) < len(self.agents):
            raise ValueError("Not enough free cells to place all agents.")
        rng.shuffle(free_cells)
        return free_cells[: len(self.agents)]

    def holes_in_the_ice(self, state, agent_name):
        """
        Returns the penalty if an agent falls into a hole.

        :param state: Agent state (must include pos_x, pos_y)
        :param agent_name: Name of the agent
        :return: Penalty or 0
        """
        agent_pos = (state["pos_x"], state["pos_y"])
        if agent_pos in self.holes:
            self.agent_fail[agent_name] = True
            return self.penalty_amount
        else:
            return 0

    def check_terminations(self):
        """
        Checks which agents should terminate or truncate.

        Termination conditions (evaluated per agent):
        - per-agent step limit (or global timestep fallback) exceeds 1000
        - agent falls into a hole (fail state)
        """
        terminations = {a.name: False for a in self.agents}
        truncations = {a.name: False for a in self.agents}

        for agente in self.agents:
            # Stop an agent that has been running too long
            if self.agent_steps.get(agente.name, 0) > 1000 or self.timestep > 1000:
                terminations[agente.name] = True
                truncations[agente.name] = True

            # Stop an agent whose Reward Machine has reached the final state
            rm = getattr(agente, "reward_machine", None)
            if rm and rm.get_current_state() == rm.get_final_state():
                terminations[agente.name] = True

            # Only the failing agent terminates if it falls into a hole
            if self.agent_fail[agente.name]:
                terminations[agente.name] = True

        return terminations, truncations

    def get_state(self, agent):
        """
        Returns a *copy* of the agent's current state to avoid accidental modification.
        """
        current_state = agent.get_state()
        return current_state.copy()

    def apply_action(self, agent, action_name: str):
        """
        Applies a deterministic action to the agent.

        Moves within grid boundaries. Holes, rewards, and stochastic behavior
        are handled elsewhere (in step()).
        """
        x, y = agent.get_position()

        if action_name == "up" and y > 0:
            y -= 1
        elif action_name == "down" and y < self.grid_height - 1:
            y += 1
        elif action_name == "left" and x > 0:
            x -= 1
        elif action_name == "right" and x < self.grid_width - 1:
            x += 1

        agent.set_position(x, y)

    def get_stochastic_action(self, agent, intended_action):
        """
        Returns a stochastic variation of the intended action.

        If delay_action is True:
            60% chance to wait
            36% chance to perform intended action
            2% chance for each perpendicular action

        Otherwise:
            80% intended action
            10% each perpendicular action
        """
        if self.delay_action:
            action_map = {
                "left": (["wait", "left", "up", "down"], [0.6, 0.36, 0.02, 0.02]),
                "right": (["wait", "right", "up", "down"], [0.6, 0.36, 0.02, 0.02]),
                "up": (["wait", "up", "left", "right"], [0.6, 0.36, 0.02, 0.02]),
                "down": (["wait", "down", "left", "right"], [0.6, 0.36, 0.02, 0.02]),
            }
        else:
            action_map = {
                "left": (["left", "up", "down"], [0.8, 0.1, 0.1]),
                "right": (["right", "up", "down"], [0.8, 0.1, 0.1]),
                "up": (["up", "left", "right"], [0.8, 0.1, 0.1]),
                "down": (["down", "left", "right"], [0.8, 0.1, 0.1]),
            }

        actions, probabilities = action_map[intended_action]
        rng = self.rng or np.random.default_rng()
        chosen_action = rng.choice(actions, p=probabilities)
        return chosen_action
