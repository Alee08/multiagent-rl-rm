import numpy as np
import random
from multiagent_rlrm.utils.utils import encode_state
from multiagent_rlrm.learning_algorithms.qlearning_lambda import QLearningLambda
from multiagent_rlrm.learning_algorithms.qlearning import QLearning
from multiagent_rlrm.multi_agent.base_environment import BaseEnvironment
from multiagent_rlrm.environments.office_world.config_office import (
    can_move_up,
    can_move_down,
    can_move_left,
    can_move_right,
)
from multiagent_rlrm.multi_agent.action_rl import ActionRL

# from building_RM import RM_dict, RM_dict_true, RM_dict_true_seq
random.seed(a=123)
np.random.seed(123)


class MultiAgentOfficeWorld(BaseEnvironment):
    """Grid-world environment for multi-agent Office World with symbolic actions and reward machines."""

    metadata = {"name": "multi_agent_office_world"}

    def __init__(
        self,
        width,
        height,
        plants,
        coffee,
        letters,
        walls,
        plants_penalty_value,
        wall_penalty_value,
        terminate_on_plants,
        terminate_hit_walls,
        all_slip=False,
    ):
        """
        Initializes the Office World environment with geometry, hazards, and control flags.

        :param width: Grid width.
        :param height: Grid height.
        :param plants: Iterable of plant coordinates.
        :param coffee: Iterable of coffee coordinates.
        :param letters: Iterable of letter coordinates.
        :param walls: Iterable of directed wall tuples ((x1, y1), (x2, y2)).
        :param plants_penalty_value: Penalty applied when stepping on a plant.
        :param wall_penalty_value: Penalty applied when hitting a wall.
        :param terminate_on_plants: Whether episodes terminate upon hitting a plant.
        :param terminate_hit_walls: Whether episodes terminate upon hitting a wall.
        :param all_slip: Whether stochastic dynamics allow slipping in all directions.
        """
        super().__init__(width, height)
        self.wait_action = ActionRL("wait", [], [])
        self.plants = plants  # Lists of plant positions in rooms
        self.coffee = coffee  # Lists of coffee positions in rooms
        self.letters = letters  # Lists of letters positions in rooms
        self.walls = walls  # Lists of walls positions in rooms
        self.terminate_on_plants = terminate_on_plants
        self.terminate_hit_walls = terminate_hit_walls
        self.possible_actions = ["up", "down", "left", "right"]
        self.rewards = 0
        self.stochastic = False  # stochastic action effects
        self.high_prob = 0.8  # prob for nominal outcome in stochastic domains
        self.plants_penalty_value = plants_penalty_value
        self.wall_penalty_value = wall_penalty_value
        self.active_agents = {agent.name: True for agent in self.agents}
        self.agent_fail = {agent.name: False for agent in self.agents}
        self.agent_steps = {agent.name: 0 for agent in self.agents}
        self.delay_action = False  # Attribute for delaying actions
        self.all_slip = all_slip
        self.map_height = height
        self.map_width = width
        self.rng = None

    def reset(self, seed=123, options=None):
        """
        Resets the environment to its initial state.

        This method resets agent rewards, states, and learning algorithms,
        while re-seeding the random number generator for reproducibility.

        :param seed: Random seed for reproducibility.
        :param options: Additional customization options (currently unused).
        :return: A tuple containing:
            - Observations: Current state for each agent.
            - Infos: Additional information (empty by default).
        """
        self.rewards = {agent.name: 0 for agent in self.agents}
        self.timestep = 0
        # self.agent_states = {agent.name: {} for agent in self.agents}
        self.active_agents = {agent.name: True for agent in self.agents}
        self.agent_fail = {agent.name: False for agent in self.agents}
        self.agent_steps = {agent.name: 0 for agent in self.agents}
        self.rng = np.random.default_rng(seed=seed)

        for agent in self.agents:
            agent.reset()  # RM reset, agent messages and status
            l_algo = agent.get_learning_algorithm()
            if isinstance(l_algo, QLearningLambda):
                l_algo.reset_e_table()  # If is ql_lambda
            initial_position = (
                agent.get_state()
            )  # Assume that get_state returns a dictionary with pos_x and pos_y

            if isinstance(l_algo, QLearning):
                l_algo.learn_done_episode()
            # self.agent_states[agent.name] = initial_position
            # agent.get_learning_algorithm().update_epsilon()
            # l_algo.learn_init()
            # l_algo.learn_init_episode()
            # l_algo.learn_done_episode()

        # observations = self.agent_states
        # Get dummy infos
        infos = {agent: {} for agent in self.agents}
        observations = {agent.name: agent.state for agent in self.agents}

        return observations, infos

    def step(self, actions):
        """
        Executes a single step in the environment.

        This function processes the actions chosen by each agent, applies penalties
        (e.g., for hitting walls), executes stochastic effects if enabled, updates
        the environment state, and calculates rewards. It also tracks termination
        and truncation conditions for each agent.

        :param actions: Dictionary of actions, where keys are agent names and values are the chosen actions.
        :return: A tuple containing:
            - Observations: Updated state for each agent.
            - Rewards: Dictionary of rewards for each agent.
            - Terminations: Dictionary indicating whether each agent's episode is terminated.
            - Truncations: Dictionary indicating whether each agent's episode is truncated.
            - Infos: Additional debug information for each agent.
        """
        self.rewards = {a.name: 0 for a in self.agents}
        infos = {a.name: {} for a in self.agents}
        for agent in self.agents:
            # TODO Is it useful? If uncommented it gives an error when using VI
            if not self.active_agents[agent.name]:
                continue
            current_statee = self.get_state(agent)

            ag_intended_action = actions[agent.name]  # action chosen by the agent

            # Calculate the penalty for walls
            wall_penalty, final_action_name = self.apply_wall_penalty(
                agent, ag_intended_action.name
            )

            # If the environment is stochastic and the action is not "wait", apply stochastic effects
            if self.stochastic and final_action_name != "wait":
                final_action_name = self.get_stochastic_action(agent, final_action_name)

            # Apply the selected action directly (no Agent.execute_action)
            self.apply_action(agent, final_action_name)

            new_state = self.get_state(agent)
            # state_rm = agent.reward_machine.get_current_state()

            # reward_rm = agent.get_reward_machine().step(new_state)
            # reward_rm = agent.get_reward_machine().get_reward(event)
            # reward = reward_rm + reward_env

            # Calculate all environmental rewards
            reward_env = self.calculate_environment_rewards(
                agent, new_state, wall_penalty
            )
            # new_state_rm = agent.reward_machine.get_current_state()
            self.rewards[agent.name] += reward_env

            # Two RM states for this agent for use in policy update
            # agent.rm_state = state_rm
            # agent.next_rm_state = new_state_rm

            # Increase step count for active agent
            self.agent_steps[agent.name] += 1
            # Update `infos` with previous and current states
            infos[agent.name]["prev_s"] = current_statee
            # infos[agent.name]["prev_q"] = state_rm
            infos[agent.name]["s"] = new_state
            # infos[agent.name]["q"] = new_state_rm
            infos[agent.name]["Renv"] = reward_env
            # infos[agent.name]["RQ"] = reward_rm
        self.timestep += 1
        # Update termination and truncation information for all agents
        terminations, truncations = self.check_terminations()

        # Update active_agents based on terminations and truncations
        for agent_name in terminations:
            if terminations[agent_name]:
                self.active_agents[agent_name] = False
        for agent_name in truncations:
            if truncations[agent_name]:
                self.active_agents[agent_name] = False

        # Return observations, rewards, terminations, truncations and updated information
        observations = {agent.name: agent.state for agent in self.agents}
        return observations, self.rewards, terminations, truncations, infos

    def plants_in_the_office(self, state, agent_name):
        """
        Checks if an agent is located on a plant and returns the penalty.

        If `terminate_on_plants` is set to True, the agent's episode is marked as failed.

        :param state: The current state of the agent, containing positional information.
        :param agent_name: The name of the agent being checked.
        :return: The penalty value if the agent is on a plant, otherwise 0.
        """

        agent_pos = (state["pos_x"], state["pos_y"])
        if agent_pos in self.plants:
            if self.terminate_on_plants:
                self.agent_fail[agent_name] = True  # Episode ends for agent
            return self.plants_penalty_value  # Penalty for hitting a plant
        return 0

    def calculate_environment_rewards(self, agent, state, wall_penalty):
        """
        Calculates all environmental rewards for a given agent.

        :param agent: The agent.
        :param state: The agent's current state.
        :param wall_penalty: The penalty incurred for colliding with walls.
        :return: The total environmental rewards, including wall penalties and other factors.
        """
        # Penalty for hitting walls
        total_reward = wall_penalty

        # Penalty or reward for other environmental elements (e.g. plants)
        total_reward += self.plants_in_the_office(state, agent.name)

        # Add additional rewards or penalties here if needed
        return total_reward

    def check_terminations(self):
        """
        Determines if the episode should terminate or truncate for each agent.

        :return: Two dictionaries:
                 - `terminations`: Indicates whether each agent's episode has terminated.
                 - `truncations`: Indicates whether each agent's episode has been truncated.
        """
        terminations = {a.name: False for a in self.agents}
        truncations = {a.name: False for a in self.agents}

        for agent in self.agents:
            if self.agent_fail[agent.name]:
                terminations[agent.name] = True
            if self.timestep > 1000:  # TODO: make this threshold configurable
                truncations[agent.name] = True

        return terminations, truncations

    def get_state(self, agent):
        """
        Retrieves the current state of the specified agent.

        :param agent: The agent whose state is being retrieved.
        :return: A copy of the agent's current state to prevent accidental modifications.
        """
        current_state = agent.get_state()
        return current_state.copy()

    def apply_action(self, agent, action_name: str):
        """
        Applies a deterministic movement to the agent based on the given action
        name. Boundary checks and wall collisions are handled using the helper
        predicates from `config_office.py`.

        :param agent: Agent to move.
        :param action_name: Name of the action ("up", "down", "left", "right", or "wait").
        """
        x, y = agent.get_position()

        if action_name == "up" and can_move_up(agent):
            y += 1
        elif action_name == "down" and can_move_down(agent):
            y -= 1
        elif action_name == "left" and can_move_left(agent):
            x -= 1
        elif action_name == "right" and can_move_right(agent):
            x += 1

        agent.set_position(x, y)

    def is_wall_collision(self, agent, action_name):
        """
        Checks whether executing the given action would result in a wall collision.

        :param agent: Agent attempting the action.
        :param action_name: Name of the intended action.
        :return: True if the action would hit a wall, otherwise False.
        """
        if action_name == "wait":
            return False  # "wait" should not cause collisions
        if action_name not in ["up", "down", "left", "right"]:
            raise ValueError(f"Invalid action: {action_name}")
        action_validity_map = {
            "up": can_move_up(agent),
            "down": can_move_down(agent),
            "left": can_move_left(agent),
            "right": can_move_right(agent),
        }
        return not action_validity_map[action_name]

    def apply_wall_penalty(self, agent, intended_action_name):
        """
        Applies a penalty if the agent's action results in a collision with a wall.
        Terminates the agent's episode if `terminate_hit_walls` is set to True.

        :param agent: The agent attempting the action.
        :param intended_action_name: The action the agent intends to perform.
        :return: A tuple (penalty, chosen_action) where chosen_action may be
                 replaced with "wait" when a collision occurs.
        """
        if self.is_wall_collision(agent, intended_action_name):
            if self.terminate_hit_walls:
                self.agent_fail[agent.name] = True  # Episode ends for the agent
            return self.wall_penalty_value, self.wait_action.name
        return 0, intended_action_name

    def get_action_probability_mapping(self):
        """
        Returns the available actions and their execution probabilities based on environment settings.

        :return: Dictionary mapping nominal action names to (subaction_list, probability_list).
        """
        if self.delay_action:
            return {
                "left": (["wait", "left", "up", "down"], [0.6, 0.36, 0.02, 0.02]),
                "right": (["wait", "right", "up", "down"], [0.6, 0.36, 0.02, 0.02]),
                "up": (["wait", "up", "left", "right"], [0.6, 0.36, 0.02, 0.02]),
                "down": (["wait", "down", "left", "right"], [0.6, 0.36, 0.02, 0.02]),
            }
        if not self.stochastic:
            return {
                "left": (["left"], [1.0]),
                "right": (["right"], [1.0]),
                "up": (["up"], [1.0]),
                "down": (["down"], [1.0]),
                "wait": (["wait"], [1.0]),  # Include if "wait" is an allowed action
            }
        if self.all_slip:  # stochastic mode with symmetric slip
            hp = self.high_prob  # 0.8
            lp = (1 - hp) / 3  # 0.066666…

            return {
                "left": (["left", "right", "up", "down"], [hp, lp, lp, lp]),
                "right": (["right", "left", "up", "down"], [hp, lp, lp, lp]),
                "up": (["up", "down", "left", "right"], [hp, lp, lp, lp]),
                "down": (["down", "up", "left", "right"], [hp, lp, lp, lp]),
            }
        else:
            hp = self.high_prob
            lp = (1 - hp) / 2
            return {
                "left": (["left", "up", "down"], [hp, lp, lp]),
                "right": (["right", "up", "down"], [hp, lp, lp]),
                "up": (["up", "left", "right"], [hp, lp, lp]),
                "down": (["down", "left", "right"], [hp, lp, lp]),
            }

    def get_stochastic_action(self, agent, intended_action_name):
        """
        Determines the stochastic outcome of the agent's intended action.

        :param agent: The agent executing the action.
        :param intended_action_name: The action the agent intends to perform.
        :return: The actual action executed, based on stochastic probabilities.
        """
        action_map = self.get_action_probability_mapping()
        actions, probabilities = action_map[intended_action_name]
        chosen_action = self.rng.choice(actions, p=probabilities)
        return chosen_action

        # -------------------------------------------------------------------------------------------
        # Generic helpers to keep get_mdp general and decoupled from x, y, q_rm
        # -------------------------------------------------------------------------------------------

    def set_state(self, agent, state):
        """
        Sets the environment to a specific state tuple (x, y, q_rm).

        :param agent: Agent whose state should be updated.
        :param state: Tuple (pos_x, pos_y, rm_state_index).
        """
        x, y, q_rm = state
        rm = agent.get_reward_machine()

        agent.set_position(x, y)
        rm.current_state = q_rm  # rm.get_state_from_index(q_rm)

    def get_current_state(self, agent):
        """
        Returns the current environment state as (x, y, q_rm).

        :param agent: Agent whose state should be read.
        :return: Tuple containing grid coordinates and reward-machine state index.
        """
        x, y = agent.get_position()
        rm = agent.get_reward_machine()
        q_rm_state = rm.get_current_state()
        q_rm = rm.state_indices[q_rm_state]
        return (x, y, q_rm)

    def is_terminal_state_mdp(self, agent, pos_x, pos_y, rm_state):
        """
        Checks whether a given MDP state is terminal.

        :param agent: Agent under evaluation.
        :param pos_x: X position.
        :param pos_y: Y position.
        :param rm_state: Reward-machine state index.
        :return: (is_terminal, terminal_reward) pair.
        """
        # 1) if plants
        if (pos_x, pos_y) in self.plants and self.terminate_on_plants:
            return True, self.plants_penalty_value

        # 2) if final state of RewardMachine
        rm = agent.get_reward_machine()
        if rm_state == rm.get_final_state():
            # Assumption: final RM state gives zero reward and terminates
            return True, 0

        # Otherwise, not terminal
        return False, 0

    def get_action_distribution(self, action):
        """
        Returns sub-actions and probabilities for a nominal ActionRL when using stochastic dynamics.

        :param action: The nominal ActionRL instance.
        :return: Tuple (subaction_objects, probabilities).
        """
        action_map = self.get_action_probability_mapping()

        subactions, probabilities = action_map[action.name]
        # Convert subaction names into actual ActionRL objects
        agent = self.agents[0]
        subaction_objects = []
        for sa_name in subactions:
            if sa_name == "wait":
                subaction_objects.append(self.wait_action)
            else:
                subaction_objects.append(agent.action(sa_name))
        return subaction_objects, probabilities
