from multiagent_rlrm.multi_agent.reward_machine import RewardMachine


class StubEventDetector:
    def __init__(self, events):
        self._events = list(events)

    def detect_event(self, _state):
        return self._events.pop(0) if self._events else None


def test_reward_machine_transitions_and_reset():
    transitions = {("q0", "a"): ("q1", 1), ("q1", "b"): ("qf", 2)}
    rm = RewardMachine(transitions, StubEventDetector(["a", "b", "b"]))

    assert rm.get_current_state() == "q0"
    assert rm.numbers_state() == 3
    assert rm.get_state_index("q0") == 0

    reward_first = rm.step(current_state=None)
    assert reward_first == 1
    assert rm.get_current_state() == "q1"

    reward_second = rm.step(current_state=None)
    assert reward_second == 2
    assert rm.get_current_state() == "qf"

    # No extra transition defined, should stay in final state with zero reward
    assert rm.step(current_state=None) == 0
    assert rm.get_current_state() == "qf"

    # Non-current state lookup does not mutate current state
    next_state, reward = rm.get_reward_for_non_current_state("q0", "a")
    assert (next_state, reward) == ("q1", 1)
    missing_state = rm.get_reward_for_non_current_state("q1", "missing")
    assert missing_state == (None, 0)

    assert rm.get_state_from_index(1) == "q1"

    rm.reset_to_initial_state()
    assert rm.get_current_state() == "q0"
