from abc import ABC, abstractmethod


class EventDetector(ABC):
    """
    Abstract base class for event detectors used by Reward Machines.

    An EventDetector takes the current environment/agent state and determines
    whether an event has occurred (e.g., reaching a position, satisfying a condition, etc.).
    """

    @abstractmethod
    def detect_event(self, current_state):
        """
        Detect an event based on the agent's current state.

        :param current_state: Dictionary representing the agent or environment state.
        :return: A detected event (any hashable type) if triggered, otherwise None.
        """
        pass
