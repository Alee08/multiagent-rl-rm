from multiagent_rlrm.multi_agent.event_detector import EventDetector


class PositionEventDetector(EventDetector):
    """
    Detects events based on the agent's current position on the grid.
    """

    def __init__(self, positions):
        """
        Initializes the detector with the set of relevant positions.

        :param positions: Iterable of tuples representing event-triggering positions.
        """
        self.positions = positions

    def detect_event(self, current_state):
        """
        Detects an event based on the agent's current position.

        :param current_state: Current agent state dictionary.
        :return: The triggering position if matched, otherwise None.
        """
        ag_current_position = (current_state["pos_x"], current_state["pos_y"])
        if ag_current_position in self.positions:
            return ag_current_position
        return None
