from unified_planning.model.multi_agent import Agent
from multiagent_rlrm.multi_agent.reward_machine import RewardMachine
from typing import Optional
from multiagent_rlrm.utils.message import Message
from multiagent_rlrm.multi_agent.state_encoder import StateEncoder
from multiagent_rlrm.multi_agent.action_encoder import ActionEncoder


class AgentRL(Agent):
    """
    Agent with Reinforcement Learning capabilities.
    """

    def __init__(
        self, name: str, ma_problem, reward_machine: Optional["RewardMachine"] = None
    ):
        """
        Initialize an RL-capable agent.

        :param name: Unique name of the agent.
        :param ma_problem: Reference to the multi-agent planning problem.
        :param reward_machine: RewardMachine instance specific to this agent.
        """
        super().__init__(name, ma_problem)
        self.reward_machine = reward_machine
        self.ma_problem = ma_problem
        self.actions_dict = {}
        self.learning_algorithm = None
        self.message_conditions = None
        self.messages = {}  # Internal state to keep track of relevant information
        self.message_sent = False
        self.position = None
        self.state = {}
        self.actions_ = []
        self.initial_position = {}
        self.initial_state = {}
        self.rm_state = None
        self.next_rm_state = None
        self.encoder: Optional[StateEncoder] = None
        self.action_encoder: Optional[ActionEncoder] = None

    def action(self, name: str) -> "up.model.action.Action":
        """
        Returns the action with the given name.

        :param name: Name of the target action.
        :return: The action with the given name in the problem.
        :raises UPValueError: If the action is not defined.
        """
        for a in self.actions_:
            if a.name == name:
                return a
        raise UPValueError(f"Action of name: {name} is not defined!")

    def add_action(self, action):
        """Add a single action to the internal action list."""
        self.actions_.append(action)

    def get_actions(self):
        """Return the list of actions of this agent."""
        return self.actions_

        # Additional RL-specific attributes can be added if needed.

    def add_state_encoder(self, encoder: StateEncoder):
        """
        Set or replace the agent's state encoder.
        """
        self.encoder = encoder

    def add_action_encoder(self, encoder: "ActionEncoder"):
        """
        Set the action encoder and let it populate self.actions_.
        """
        self.action_encoder = encoder
        encoder.build_actions()

    def select_action(self, state, best=False):
        """
        Select an action to perform based on the current state and Reward Machine state.

        :param state: Current state of the agent (dictionary).
        :param best: If True, select the best action (e.g., greedy, no epsilon exploration).
        :return: The selected action object.
        :raises Exception: If the state encoder is not set.
        """
        if not self.encoder:
            raise Exception(
                "Encoder not set. Please add an encoder before selecting actions."
            )
        encoded_state, info = self.encoder.encode(state)

        # Select the action index from the learning algorithm
        action_index = self.get_learning_algorithm().choose_action(
            encoded_state, best, info=info
        )

        # Retrieve the corresponding action from the internal dictionary
        action = self.actions_dix()[action_index]
        if action is None:
            raise ValueError(
                f"Action index {action_index} not found in actions dictionary."
            )
        return action

    def update_policy(self, state, action, reward, next_state, terminated, **kwargs):
        """
        Update the agent's policy using the underlying learning algorithm.

        :param state: Current state.
        :param action: Executed action.
        :param reward: Scalar reward signal.
        :param next_state: Next state after executing the action.
        :param terminated: Boolean flag indicating if the episode terminated.
        :param kwargs: Extra info, typically including RM and environment info:
                       - infos (dict): may contain prev_q, q, Renv, RQ, qrm_experience, reward_machine.
        :return: Whatever the learning algorithm returns to indicate termination request.
        :raises Exception: If encoder or reward machine are not set.
        """
        # Extract infos if provided
        infos = kwargs.get("infos", {})

        # Ensure encoder is set
        if not self.encoder:
            raise Exception(
                "Encoder not set. Please add an encoder before updating policy."
            )
        # Ensure Reward Machine is set
        if not self.reward_machine:
            raise Exception(
                "Reward Machine not set. Cannot update policy without Reward Machine."
            )

        # Values from infos, falling back to defaults when missing
        state_rm = infos.get("prev_q", 0)
        next_state_rm = infos.get("q", 0)
        reward_env = infos.get("Renv", 0)
        reward_q = infos.get("RQ", 0)
        qrm_experience = infos.get("qrm_experience", [])
        reward_machine = infos.get("reward_machine", [])

        # Encode current and next states with RM state
        encoded_current_state, current_info = self.encoder.encode(state, state_rm)
        encoded_next_state, next_info = self.encoder.encode(next_state, next_state_rm)

        # Convert Reward Machine states to indices if they are not zero
        state_rm_idx = (
            self.reward_machine.get_state_index(state_rm) if state_rm != 0 else 0
        )
        next_state_rm_idx = (
            self.reward_machine.get_state_index(next_state_rm)
            if next_state_rm != 0
            else 0
        )

        # Build the info dictionary to pass to the learning algorithm
        info = {
            "prev_s": current_info["s"],
            "s": next_info["s"],
            "prev_q": state_rm_idx,
            "q": next_state_rm_idx,
            "Renv": reward_env,
            "RQ": reward_q,
            "qrm_experience": qrm_experience,
            "reward_machine": reward_machine,
        }

        action_index = self.actions_idx(action)

        # Update learning algorithm
        term = self.get_learning_algorithm().update(
            encoded_current_state,
            encoded_next_state,
            action_index,
            reward,
            terminated,
            info=info,
        )

        # Whether the algorithm requests to terminate
        return term

    def actions_dix(self):
        """
        Build (or rebuild) the mapping from action index to action object.

        :return: Dictionary {index: action}.
        """
        for idx, act in enumerate(self.actions_):
            self.actions_dict[idx] = act
        return self.actions_dict

    def actions_idx(self, action):
        """
        Retrieve the index associated with a given action object.

        :param action: Action object to look up.
        :return: Index of the action in the internal dictionary (or None if not found).
        """
        dict_ = self.actions_dix()
        found_key = None
        for key, value in dict_.items():
            if value == action:
                found_key = key
                break

        return found_key

    def get_reward(self, event):
        """
        Get the reward from the Reward Machine for a given event.

        :param event: Event (e.g., label) to query the Reward Machine.
        :return: Reward value (0 if no Reward Machine is set).
        """
        return self.reward_machine.get_reward(event) if self.reward_machine else 0

    def set_reward_machine(self, reward_machine: RewardMachine):
        """Assign a Reward Machine to this agent."""
        self.reward_machine = reward_machine

    def get_reward_machine(self):
        """Return the agent's Reward Machine."""
        return self.reward_machine

    def add_rl_action(self, action):
        """
        Add an RL-specific action to the agent.

        :param action: RL action to be added.
        """
        # Implementation depends on the structure of RL actions
        self.actions_.append(action)

    # Other RL-specific methods can be added here

    def set_learning_algorithm(self, algorithm):
        """
        Assign a learning algorithm to the agent.

        :param algorithm: Instance of the learning algorithm.
        """
        self.learning_algorithm = algorithm

    def get_learning_algorithm(self):
        """
        Return the agent's learning algorithm.
        """
        return self.learning_algorithm

    def _send_message(self, agents, condition):
        """
        Send a message to other agents when this agent reaches a certain condition/state.

        :param agents: List of recipient agents.
        :param condition: Condition to communicate to the other agents.
        """
        message = Message(self.name, condition)
        # Logic to broadcast the message to other agents
        self.ma_problem.broadcast_message(agents, message)

    def _receive_message(self, message):
        """
        Handle an incoming message.

        :param message: Message instance received from another agent.
        """
        if isinstance(message, Message):
            # Update internal state based on the message
            self.process_message(message)

    def process_message(self, message):
        """
        Process a received message and update internal structures accordingly.

        :param message: Message instance.
        """
        # Create a key (sender, fluent) from the message content
        key = (
            message.sender,
            message.condition[0][0],
        )
        self.messages[key] = message.condition[0][1]
        # self.messages[message.sender] = message.condition

        # Example logic for specific message types
        """
        if message.condition == some_specific_condition:
            self.take_specific_action()
        """

    def take_specific_action(self):
        """
        Implement specific actions to take in response to certain messages or conditions.
        """
        pass

    def reset_messages(self):
        """Clear all stored messages and message-related conditions."""
        self.messages = {}
        self.message_conditions = None

    def return_messages(self):
        """Return the internal message dictionary."""
        return self.messages

    def execute_action(self, action):
        """
        Execute an action if all its preconditions are satisfied.

        :param action: Action object to be executed.
        :return: True if the action is executed, False otherwise.
        :raises ValueError: If the action is None.
        """
        if action is None:
            raise ValueError(f"Action {action.name} not found for agent {self.name}.")

        if all(pre(self) for pre in action.preconditions):
            for effect in action.effects:
                effect(self)
            return True
        else:
            # Preconditions not satisfied; action is not executed.
            return False

    def set_initial_position(self, pos_x, pos_y):
        """
        Store and set the agent's initial position.

        :param pos_x: X coordinate.
        :param pos_y: Y coordinate.
        """
        self.initial_position = (pos_x, pos_y)
        self.set_position(pos_x, pos_y)

    def set_position(self, pos_x, pos_y):
        """
        Set the agent's position and update its state accordingly.

        :param pos_x: X coordinate.
        :param pos_y: Y coordinate.
        """
        self.position = (pos_x, pos_y)
        self.add_to_state("pos_x", pos_x)
        self.add_to_state("pos_y", pos_y)

    def get_position(self):
        """Return the current (x, y) position of the agent."""
        return self.position

    def add_to_state(self, key, value):
        """
        Add/update an attribute in the current state and in the initial state.

        :param key: State attribute name.
        :param value: Value to set for this attribute.
        """
        self.state[key] = value
        self.initial_state[key] = value  # Also store in the initial-state snapshot

    def reset(self):
        """
        Reset the agent's position and state to the initial configuration.

        Also clears messages and resets the Reward Machine to its initial state (if any).
        """
        # Use the stored initial position
        self.set_position(*self.initial_position)
        # Reset state to initial attributes
        self.state = self.initial_state.copy()
        self.reset_messages()
        if self.reward_machine:
            self.reward_machine.reset_to_initial_state()

    def set_state(self, **kwargs):
        """
        Set arbitrary attributes in the agent's state.

        :param kwargs: Mapping of attribute names to their values.
        """
        for key, value in kwargs.items():
            self.state[key] = value

    def get_state(self):
        """
        Return the current state of the agent.

        :return: Dictionary representing the agent's state.
        """
        return self.state
