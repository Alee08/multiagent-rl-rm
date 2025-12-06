from multiagent_rlrm.environments.frozen_lake.state_encoder_frozen_lake import (
    StateEncoderFrozenLake,
)
from multiagent_rlrm.multi_agent.reward_machine import RewardMachine


class DummyEventDetector:
    def detect_event(self, _state):
        return None


class DummyAgent:
    def __init__(self, width, height, rm):
        self.ma_problem = type("P", (), {"grid_width": width, "grid_height": height})
        self._rm = rm

    def get_reward_machine(self):
        return self._rm


def test_state_encoder_frozen_lake_encode():
    transitions = {("q0", None): ("q0", 0)}
    rm = RewardMachine(transitions, DummyEventDetector())
    agent = DummyAgent(width=4, height=5, rm=rm)
    encoder = StateEncoderFrozenLake(agent)

    encoded, info = encoder.encode({"pos_x": 1, "pos_y": 2}, state_rm="q0")

    assert encoded == 9  # pos_index=2*4+1=9, single RM state
    assert info["s"] == 9
    assert info["q"] == 0
