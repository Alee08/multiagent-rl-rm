from abc import ABC, abstractmethod


class EventDetector(ABC):
    @abstractmethod
    def detect_event(self, current_state):
        pass
