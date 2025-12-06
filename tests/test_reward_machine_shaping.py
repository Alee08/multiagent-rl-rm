from multiagent_rlrm.multi_agent.reward_machine import RewardMachine


class NoopEventDetector:
    def detect_event(self, _state):
        return None


def test_add_distance_reward_shaping():
    transitions = {("q0", "a"): ("q1", 0), ("q1", "b"): ("qf", 1)}
    rm = RewardMachine(transitions, NoopEventDetector())

    rm.add_distance_reward_shaping(gamma=0.9, rs_gamma=0.9, alpha=5)

    assert rm.potentials["qf"] == 0  # final state
    assert rm.potentials["q1"] == -5  # one step away
    assert rm.potentials["q0"] == -10  # two steps away
