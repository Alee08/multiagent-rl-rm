from abc import ABC, abstractmethod


class StateEncoder(ABC):
    def __init__(self, agent):
        self.agent = agent  # Memorizza un riferimento all'agente

    @abstractmethod
    def encode(self, state, state_rm=None):
        """
        Codifica lo stato base, da estendere nelle classi derivate.
        :param state: Dizionario che rappresenta lo stato dell'agente.
        :return: Codifica numerica dello stato.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def encode_rm_state(self, state_rm):
        """
        Encodes the Reward Machine state.
        """
        if state_rm is None:
            rm_state = self.agent.get_reward_machine().get_current_state()
            rm_state_index = self.agent.get_reward_machine().get_state_index(rm_state)
        else:
            rm_state_index = self.agent.get_reward_machine().get_state_index(state_rm)
        return rm_state_index
