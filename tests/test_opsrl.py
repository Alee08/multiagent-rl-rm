import numpy as np

from multiagent_rlrm.learning_algorithms.opsrl import OPSRL


def test_opsrl_counts_and_policy_update():
    algo = OPSRL(
        state_space_size=2,
        action_space_size=1,
        ep_len=3,
        gamma=0.9,
        prior_transition="uniform",
        seed=123,
    )

    before_dirichlet = algo.N_sas.copy()
    before_beta = algo.M_sa.copy()

    done = algo.update(
        encoded_state=0,
        encoded_next_state=1,
        action=0,
        reward=0.8,
        terminated=True,
    )

    assert done is True
    # Counts should have increased
    assert np.all(algo.N_sas[0, 0] >= before_dirichlet[0, 0])
    assert np.all(algo.M_sa[0, 0] >= before_beta[0, 0])
    # MAP policy computed
    assert algo.Q_policy.shape[0] == algo.H
    assert algo.greedy_action(state=0, hh=0) in (0, 0)  # returns int 0
