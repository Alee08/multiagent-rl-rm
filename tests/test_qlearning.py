import numpy as np

from multiagent_rlrm.learning_algorithms.qlearning import QLearning


def test_qlearning_update_greedy():
    ql = QLearning(
        gamma=0.9,
        action_selection="greedy",
        learning_rate=1.0,
        state_space_size=2,
        action_space_size=2,
        qtable_init=1.0,
    )

    # Update on (s=0, a=0, r=1, s'=1)
    ql.update(
        encoded_state=0,
        encoded_next_state=1,
        action=0,
        reward=1.0,
        terminated=False,
        info={},
    )
    # With lr=1: Q = r + gamma * max(Q(s'))
    expected = 1.0 + 0.9 * 1.0
    assert np.isclose(ql.q_table[0, 0], expected)


def test_qlearning_epsilon_decay_and_choice():
    ql = QLearning(
        gamma=0.9,
        action_selection="greedy",
        learning_rate=0.5,
        state_space_size=1,
        action_space_size=2,
        epsilon_start=0.5,
        epsilon_end=0.1,
        epsilon_decay=0.5,
    )
    ql.q_table[0] = np.array([0.2, 0.8])

    # best=True should ignore epsilon and pick argmax
    assert ql.choose_action(encoded_state=0, best=True) == 1

    # After one episode, epsilon decays but not below epsilon_end
    ql.learn_done_episode()
    assert np.isclose(ql.epsilon, 0.25)
    ql.learn_done_episode()
    assert np.isclose(ql.epsilon, 0.125)
    ql.learn_done_episode()
    # Should floor at epsilon_end=0.1
    assert np.isclose(ql.epsilon, 0.1)
