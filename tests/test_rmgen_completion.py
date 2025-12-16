from multiagent_rlrm.rmgen.completion import complete_missing_transitions
from multiagent_rlrm.rmgen.spec import RMSpec


def make_partial_spec():
    return RMSpec.from_dict(
        {
            "name": "partial",
            "env_id": "env",
            "version": "1.0",
            "states": ["q0", "q1"],
            "initial_state": "q0",
            "terminal_states": ["q1"],
            "event_vocabulary": ["e1", "e2"],
            "transitions": [
                {"from_state": "q0", "event": "e1", "to_state": "q1", "reward": 1}
            ],
        }
    )


def test_completion_full_cartesian():
    spec = make_partial_spec()
    spec, report = complete_missing_transitions(spec, default_reward=0.0)
    assert report["added"] == 3
    assert len(spec.transitions) == 4  # 2 states * 2 events
    # Ensure explicit transition preserved
    assert any(
        t.from_state == "q0"
        and t.event == "e1"
        and t.to_state == "q1"
        and t.reward == 1
        for t in spec.transitions
    )


def test_completion_default_reward_applied():
    spec = make_partial_spec()
    spec, _ = complete_missing_transitions(spec, default_reward=0.5)
    # A completed transition should carry the default reward
    completed = next(
        t for t in spec.transitions if t.from_state == "q1" and t.event == "e2"
    )
    assert completed.reward == 0.5


def test_terminal_self_loop_added():
    spec = make_partial_spec()
    spec, _ = complete_missing_transitions(
        spec, default_reward=0.0, terminal_self_loop=True
    )
    assert any(
        t.from_state == "q1" and t.event == "e1" and t.to_state == "q1"
        for t in spec.transitions
    )
