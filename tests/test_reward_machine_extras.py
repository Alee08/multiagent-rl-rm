from multiagent_rlrm.multi_agent.reward_machine import RewardMachine


class DummyDetector:
    def __init__(self, events):
        self.events = list(events)

    def detect_event(self, _state):
        return self.events.pop(0) if self.events else None


def test_reward_machine_unreachable_distance_and_value_iteration_relative_delta():
    transitions = {("q0", "a"): ("q1", 0)}
    rm = RewardMachine(transitions, DummyDetector(["a"]))

    # get_distance should return large number for unreachable final state
    unreachable = rm.get_distance("q_missing")
    assert unreachable == 999999

    # value_iteration with delta_rel=True should converge and keep terminal value at 0
    U = list(rm.state_indices.keys())
    delta_u = rm.get_delta_u()
    delta_r = rm.get_delta_r()
    V = rm.value_iteration(U, delta_u, delta_r, rm.get_final_state(), gamma=0.9)
    assert V[rm.get_final_state()] == 0
