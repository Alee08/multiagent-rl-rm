from multiagent_rlrm.multi_agent.event_detector import EventDetector


class PositionEventDetector(EventDetector):
    """
    Detects events based on the agent's current position.
    """

    def __init__(self, positions):
        """
        Initializes the event detector with a set of relevant positions.

        :param positions: A set of (x, y) tuples representing positions
                          that should trigger an event when reached.
        """
        self.positions = positions

    def detect_event(self, current_state):
        """
        Detects whether the agent's current position corresponds to
        a predefined event-triggering location.

        :param current_state: Dictionary representing the agent's current state.
                              Expected keys: "pos_x", "pos_y".
        :return: The position tuple if an event is detected; otherwise None.
        """
        ag_current_position = (current_state["pos_x"], current_state["pos_y"])

        # Check whether the agent stands on a position that triggers an event
        if ag_current_position in self.positions:
            return ag_current_position

        return None
