import numpy as np

from multiagent_rlrm.learning_algorithms.ucbvi import UCBVI


def test_ucbvi_update_and_planning():
    algo = UCBVI(
        state_space_size=2,
        action_space_size=1,
        ep_len=2,
        gamma=1.0,
        bonus_scaling=1.0,
        bonus_type="hoeffding",
        seed=42,
    )

    # Single transition ends episode -> triggers planning
    done = algo.update(
        encoded_state=0,
        encoded_next_state=1,
        action=0,
        reward=1.0,
        terminated=True,
    )

    assert done is True
    # Value function should be optimistic but finite
    assert algo.V_policy[0, 0] > 0
    assert algo.V_policy[0, 0] <= algo.v_max[0]
    # Choose best action returns valid index
    assert algo.choose_action(encoded_state=0, best=True) == 0
