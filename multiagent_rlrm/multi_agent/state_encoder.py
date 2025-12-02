from abc import ABC, abstractmethod


class StateEncoder(ABC):
    """Abstract encoder that maps environment states (and optional RM state) into numeric representations."""

    def __init__(self, agent):
        # Store a reference to the agent using this encoder
        self.agent = agent

    @abstractmethod
    def encode(self, state, state_rm=None):
        """
        Encode the base state. Must be extended/implemented in subclasses.

        :param state: Dictionary representing the agent's state.
        :param state_rm: Optional Reward Machine state associated with this environment state.
        :return: A numerical encoding of the state (possibly including RM state).
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def encode_rm_state(self, state_rm):
        """
        Encode the Reward Machine state into an integer index.

        If state_rm is None, the current RM state of the agent is used.

        :param state_rm: Optional Reward Machine state label or id.
        :return: Integer index of the RM state.
        """
        if state_rm is None:
            rm_state = self.agent.get_reward_machine().get_current_state()
            rm_state_index = self.agent.get_reward_machine().get_state_index(rm_state)
        else:
            rm_state_index = self.agent.get_reward_machine().get_state_index(state_rm)
        return rm_state_index
