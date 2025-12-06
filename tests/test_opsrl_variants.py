import numpy as np

from multiagent_rlrm.learning_algorithms.opsrl import OPSRL


def test_opsrl_stage_dependent_optimistic_absorbing():
    algo = OPSRL(
        state_space_size=2,
        action_space_size=1,
        ep_len=2,
        gamma=0.9,
        prior_transition="optimistic",
        stage_dependent=True,
        make_absorbing_on_done=True,
        absorb_alpha=5.0,
        seed=0,
    )

    done = algo.update(
        encoded_state=0,
        encoded_next_state=1,
        action=0,
        reward=0.5,
        terminated=True,
    )

    assert done is True
    # absorbing state should be enforced with large self-loop mass
    assert bool(algo.absorbing[1]) is True
    assert np.all(algo.N_sas[:, 1, :, 1] == algo.absorb_alpha)
    # MAP policy computed
    assert algo.Q_policy.shape[0] == algo.H
    assert algo.greedy_action(state=0, hh=0) == 0
