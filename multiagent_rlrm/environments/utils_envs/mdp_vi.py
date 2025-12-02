from config_office import config, get_experiment_for_map
import numpy as np
import copy


def value_iteration(P, num_states, num_actions, gamma=0.9, theta=1e-3, delta_rel=False):
    """
    Performs Value Iteration on a given MDP to compute the optimal value function
    and policy, supporting both absolute and relative convergence thresholds.

    Parameters:
    - P: Transition and reward model, where P[s][a] = [(prob, s_next, reward, done), ...].
    - num_states: The total number of states.
    - num_actions: The total number of actions.
    - gamma: Discount factor for future rewards.
    - theta: Convergence threshold for the value function updates.
    - delta_rel: If True, uses a relative convergence criterion.

    Returns:
    - V: Optimal state-value function.
    - policy: Optimal deterministic policy (policy[s] = best action in state s).
    - Q: Optimal Q-table (state-action value function).
    """
    # Initialize values to zero
    V = np.zeros(num_states)
    Q = np.zeros((num_states, num_actions))

    while True:
        delta = 0
        old_V = V.copy()  # Keep previous values for convergence check
        for s in range(num_states):
            q_values = np.zeros(num_actions)
            # Compute Q(s, a) for each action
            for a in range(num_actions):
                for (prob, s_next, reward, done) in P[s][a]:
                    q_values[a] += prob * (reward + gamma * V[s_next] * (not done))

            max_q = np.max(q_values)
            Q[s] = q_values

            # Calcolo differenza tra vecchio e nuovo valore per valutare convergenza
            if delta_rel:
                diff = abs(max_q - old_V[s])
                denom = abs(max_q) if abs(max_q) > 1e-12 else 1.0
                cdelta = diff / denom
            else:
                cdelta = abs(max_q - old_V[s])

            delta = max(delta, cdelta)
            V[s] = max_q

        if delta < theta:
            break

    # Calcoliamo la policy ottima.
    policy = np.argmax(Q, axis=1)

    return V, policy, Q
