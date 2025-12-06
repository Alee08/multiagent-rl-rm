import numpy as np

from multiagent_rlrm.learning_algorithms.rmax import RMax


def test_rmax_value_iteration_triggers_at_threshold():
    rmax = RMax(
        state_space_size=2,
        action_space_size=1,
        gamma=0.9,
        s_a_threshold=1,
        max_reward=1.0,
    )
    optimistic = rmax.get_q_value(0, 0)
    assert optimistic > 0

    # One observation reaches threshold -> value_iteration runs with zero reward
    rmax.update(state=0, next_state=0, action=0, reward=0.0, terminated=False)

    updated = rmax.get_q_value(0, 0)
    assert 0 <= updated < optimistic
    # Unvisited state-action remains optimistic
    assert rmax.get_q_value(1, 0) == optimistic


def test_rmax_applies_qrm_experiences_when_enabled():
    rmax = RMax(
        state_space_size=2,
        action_space_size=1,
        gamma=0.9,
        s_a_threshold=2,
        max_reward=1.0,
        use_qrm=True,
    )

    qrm_exp = (1, 0, 0.5, 1, False, None, None, None, None, None)
    rmax.update(
        state=0,
        next_state=0,
        action=0,
        reward=0.0,
        terminated=False,
        info={"qrm_experience": [qrm_exp]},
    )

    # Both main and qrm experience should increment counts
    assert rmax.s_a_counts[0, 0] == 1
    assert rmax.s_a_counts[1, 0] == 1
