from multiagent_rlrm.learning_algorithms.ucbvi import UCBVI


def test_ucbvi_stage_dependent_bernstein_bonus_and_policy():
    algo = UCBVI(
        state_space_size=2,
        action_space_size=1,
        ep_len=3,
        gamma=0.9,
        bonus_scaling=1.0,
        bonus_type="bernstein",
        stage_dependent=True,
        seed=1,
    )

    # One transition triggers episode end and planning
    done = algo.update(
        encoded_state=0,
        encoded_next_state=1,
        action=0,
        reward=0.2,
        terminated=True,
    )
    assert done is True

    # Bonus computed for first timestep
    assert algo.B_sa[0, 0, 0] > 0
    assert algo.B_sa[0, 0, 0] <= algo.v_max[0]

    # Greedy policy should be computable
    assert algo.greedy_action(state=0, hh=0) == 0
    assert algo.choose_action(encoded_state=0, best=True) == 0
