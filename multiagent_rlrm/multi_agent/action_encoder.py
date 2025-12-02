from abc import ABC, abstractmethod


class ActionEncoder(ABC):
    """Base class for domain-specific action encoders."""

    def __init__(self, agent):
        self.agent = agent

    @abstractmethod
    def build_actions(self):
        """
        Register the agent's actions for this domain,
        typically via self.agent.add_action(...).
        """
        raise NotImplementedError

    @property
    def action_names(self):
        return [a.name for a in self.agent.actions_]
