import numpy as np

from multiagent_rlrm.learning_algorithms.qlearning_lambda import QLearningLambda


def test_qlearning_lambda_update_and_traces_decay():
    ql = QLearningLambda(
        gamma=0.9,
        lambd=0.5,
        action_selection="greedy",
        learning_rate=1.0,
        state_space_size=2,
        action_space_size=1,
    )

    # Initial update; next_action greedy -> traces should decay by gamma*lambda
    ql.update(
        encoded_state=0,
        encoded_next_state=1,
        action=0,
        reward=1.0,
        terminated=False,
        next_action=0,
    )

    assert np.isclose(ql.q_table[0, 0], 1.0)  # reward + gamma*0
    assert np.isclose(ql.e_table[0, 0], 0.9 * 0.5)  # replacing trace then decay


def test_qlearning_lambda_reset_traces():
    ql = QLearningLambda(
        gamma=0.9,
        lambd=0.5,
        action_selection="greedy",
        learning_rate=0.5,
        state_space_size=1,
        action_space_size=1,
    )
    ql.e_table[0, 0] = 0.7
    ql.learn_init_episode()
    assert ql.e_table[0, 0] == 0.0
