import pytest

from multiagent_rlrm.rmgen.spec import RMSpec


BASE_SPEC = {
    "name": "reward_test",
    "env_id": "env",
    "version": "1.0",
    "states": ["q0", "q1"],
    "initial_state": "q0",
    "terminal_states": ["q1"],
    "event_vocabulary": ["e"],
}


def build_spec_with_reward(reward_value):
    data = {
        **BASE_SPEC,
        "transitions": [
            {"from_state": "q0", "event": "e", "to_state": "q1", "reward": reward_value}
        ],
    }
    return RMSpec.from_dict(data)


@pytest.mark.parametrize(
    "reward_value, expected", [("r0", 0.0), ("r1", 1.0), ("r0.5", 0.5), ("1", 1.0)]
)
def test_reward_string_with_r_prefix_and_numeric(reward_value, expected):
    spec = build_spec_with_reward(reward_value)
    assert spec.transitions[0].reward == expected


def test_invalid_reward_string_raises():
    with pytest.raises(ValueError):
        build_spec_with_reward("not_a_number")


def test_env_id_normalized_lower():
    data = {
        **BASE_SPEC,
        "env_id": "OfficeWorld ",
        "transitions": [
            {"from_state": "q0", "event": "e", "to_state": "q1", "reward": 0}
        ],
    }
    spec = RMSpec.from_dict(data)
    assert spec.env_id == "officeworld"
