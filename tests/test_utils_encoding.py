import pytest

from multiagent_rlrm.utils import utils
from multiagent_rlrm.multi_agent.reward_machine import RewardMachine


class DummyEventDetector:
    def detect_event(self, _state):
        return None


class DummyAgent:
    def __init__(self, width=2, height=2, max_time=3):
        self.ma_problem = type(
            "P", (), {"grid_width": width, "grid_height": height, "max_time": max_time}
        )
        transitions = {("q0", None): ("q0", 0)}
        self._rm = RewardMachine(transitions, DummyEventDetector())

    def get_reward_machine(self):
        return self._rm


def test_encode_state_and_time_variants():
    agent = DummyAgent()
    state = {"pos_x": 1, "pos_y": 1, "timestamp": 1, "timestep": 2}

    enc = utils.encode_state(agent, state, "q0")
    assert enc == 3  # pos_index=3, rm_idx=0

    enc_time = utils.encode_state_with_time(agent, state, "q0")
    assert enc_time == 3 * agent.ma_problem.max_time + 1

    enc_time_step = utils.encode_state_time(agent, state, "q0")
    assert enc_time_step == (3 * agent.ma_problem.max_time) + state["timestep"]


def test_encode_state_raises_on_out_of_bounds():
    agent = DummyAgent(width=1, height=1, max_time=1)
    state = {"pos_x": 0, "pos_y": 0, "timestamp": 0, "timestep": 0}

    with pytest.raises(ValueError):
        utils.encode_state_with_time(agent, state | {"timestamp": 2}, "q0")

    with pytest.raises(ValueError):
        utils.encode_state_time(agent, state | {"timestep": 2}, "q0")


def test_parse_map_emoji_and_office():
    holes, goals, dims = utils.parse_map_emoji(
        """
        🟩 🟩
        ⛔ 1
        """
    )
    assert holes == [(0, 1)]
    assert goals == {"1": (1, 1)}
    assert dims == (2, 2)

    office = """
    🟩 🪴
    🥤 ✉️
    """
    coords, goals_office, walls = utils.parse_office_world(office)
    assert coords["plant"] == [(1, 0)]
    assert coords["coffee"] == [(0, 1)]
    assert coords["letter"] == [(1, 1)]
    assert goals_office == {}
    assert isinstance(walls, list)
