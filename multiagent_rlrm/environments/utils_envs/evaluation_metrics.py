import numpy as np
import copy
import wandb
import os, math
from scipy.stats import ttest_ind


def calcola_media_mobile(valori, finestra):
    """
    Calculates a moving average over the provided values with a given window size.

    Parameters:
      - valori: Array of numerical values.
      - finestra: Size of the moving average window.

    Returns:
      - medie: Array of moving averages.
    """
    medie = np.convolve(valori, np.ones(finestra) / finestra, mode="valid")
    return medie


def test_policy_optima(
    env,
    episodi_test=100,
    window_size=100,
    optimal_steps=30,
    gamma=0.9,
    test_deterministic=None,
):
    """
    Tests an optimal policy over multiple episodes in the given environment.

    Parameters:
      - env: The environment for testing.
      - episodi_test: Number of test episodes.
      - window_size: Size of the moving average window for success rate.
      - optimal_steps: Number of optimal steps used for ARPS calculation.
      - gamma: Discount factor for rewards.
      - test_deterministic: If True, forces the environment to deterministic mode.

    Returns:
      - Success rates, moving averages, and performance metrics for the policy.
    """

    env_test = copy.deepcopy(env)  # env copy for testing
    if test_deterministic is not None:
        try:
            env_test.env.stochastic = (
                not test_deterministic
            )  # test on deterministic env
            if test_deterministic:
                episodi_test = 1
        except:
            print("Environment does not support setting stochastic mode")

    # print(f"test optimal policy: {env_test.env.stochastic} {episodi_test}")

    successi_per_agente = {ag.name: 0 for ag in env_test.agents}
    successi_per_episodio = {ag.name: [] for ag in env_test.agents}
    ricompense_per_agente = {
        ag.name: [] for ag in env_test.agents
    }  # Track cumulative rewards per episode
    timestep_per_episodio = {
        ag.name: [] for ag in env_test.agents
    }  # Track timesteps per episode
    arps_per_agente = {ag.name: [] for ag in env_test.agents}  # Track ARPS per agent

    for episodio in range(episodi_test):
        states, infos = env_test.reset(
            10000 + episodio
        )  # Reset the test environment for each episode
        done = {ag.name: False for ag in env_test.agents}
        agent_success = {ag.name: False for ag in env_test.agents}
        timestep = 0  # Initialize the timestep counter for the episode
        episode_rewards = {ag.name: 0 for ag in env_test.agents}

        cum_gamma = 1.0

        while not all(done.values()):
            actions = {}
            for ag in env_test.agents:
                current_state = env_test.env.get_state(ag)
                azione = ag.select_action(
                    current_state, best=True
                )  # Select the greedy action during testing
                # print(f"Current State: {current_state}, Select Action: {azione.name}")
                actions[ag.name] = azione

            new_states, rewards, done, truncations, infos = env_test.step(
                actions
            )  # Esegui un passo per tutti gli agenti

            for ag in env_test.agents:
                if not agent_success[ag.name]:
                    episode_rewards[ag.name] += (
                        cum_gamma * rewards[ag.name]
                    )  # Sum discounted reward
                    if (
                        done[ag.name]
                        and ag.get_reward_machine().get_current_state()
                        == ag.get_reward_machine().get_final_state()
                    ):
                        successi_per_agente[
                            ag.name
                        ] += (
                            1  # Count success when the agent reaches the final RM state
                        )
                        agent_success[ag.name] = True

            states = copy.deepcopy(new_states)  # Update state for the next timestep
            cum_gamma *= gamma

            timestep += 1  # Increment timestep counter for the episode

            if all(done.values()) or all(truncations.values()) or timestep > 1000:
                break

        # end one episode

        for ag in env_test.agents:
            successi_per_episodio[ag.name].append(1 if agent_success[ag.name] else 0)
            ricompense_per_agente[ag.name].append(episode_rewards[ag.name])
            if agent_success[ag.name]:
                timestep_per_episodio[ag.name].append(timestep)
                # Record timestep count for this successful episode

        # TODO CHECK - episode_rewards now contains discounted rewards
        # Compute discounted ARPS per episode
        for ag_name, reward in episode_rewards.items():
            discounted_reward = reward
            # sum(
            # reward * (gamma ** t) for t, reward in enumerate(rewards.values())
            # )
            if timestep > 0:  # Avoid division by zero
                arps = (discounted_reward / timestep) / optimal_steps
                arps_per_agente[ag_name].append(arps)

    # end all episodes

    success_rate_per_agente = {
        ag_name: (successi / episodi_test) * 100
        for ag_name, successi in successi_per_agente.items()
    }

    # Compute moving averages for each agent
    moving_averages = {}
    for ag_name in successi_per_episodio:
        moving_averages[ag_name] = (
            np.convolve(successi_per_episodio[ag_name], np.ones(window_size), "valid")
            / window_size
        )

    average_timesteps = -1

    if len(timestep_per_episodio[ag.name]) > 0:
        # timesteps average for successful episodes
        # print(timestep_per_episodio[ag.name])
        avg_timesteps_per_agente = {
            ag.name: np.mean(timestep_per_episodio[ag.name]) for ag in env_test.agents
        }
        std_timesteps_per_agente = {
            ag_name: np.std(timestep_per_episodio[ag.name]) for ag in env_test.agents
        }
    else:
        avg_timesteps_per_agente = {ag.name: 0 for ag in env_test.agents}
        std_timesteps_per_agente = {ag_name: 0 for ag in env_test.agents}

    # Compute average reward for each agent
    avg_reward_per_agente = {
        ag_name: np.mean(rewards) for ag_name, rewards in ricompense_per_agente.items()
    }
    std_reward_per_agente = {
        ag_name: np.std(rewards) for ag_name, rewards in ricompense_per_agente.items()
    }

    # Compute average ARPS for each agent
    avg_arps_per_agente = {
        ag_name: np.mean(arps) for ag_name, arps in arps_per_agente.items()
    }

    return (
        success_rate_per_agente,
        moving_averages,
        avg_timesteps_per_agente,
        std_timesteps_per_agente,
        avg_reward_per_agente,
        std_reward_per_agente,
        avg_arps_per_agente,  # Aggiungi ARPS al ritorno
    )


def save_q_tables(agents, directory="data"):
    """
    Saves the Q-tables of all agents into a compressed file.

    Parameters:
      - agents: List of agents with Q-tables.
      - directory: Directory where the Q-tables will be saved.
    """
    q_tables_dict = {}
    for agent in agents:
        algorithm = agent.get_learning_algorithm()
        if hasattr(algorithm, "q_table"):
            q_table = algorithm.q_table  # Shape: (nS, nQ, nA)
        elif hasattr(algorithm, "Q"):
            q_table = algorithm.Q  # Shape: (nS, nQ, nA)
        else:
            print(f"No Q-table or knownTSQA found for agent {agent.name}")
            continue
        q_tables_dict[f"q_table_{agent.name}"] = q_table
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.savez_compressed(f"{directory}/q_tables.npz", **q_tables_dict)


def update_actions_log(actions_log, actions, episode):
    """
    Updates the log of actions taken by agents during a specific episode.

    Parameters:
      - actions_log: Dictionary containing the logs for each episode.
      - actions: Dictionary of actions executed by agents in the current step.
      - episode: The current episode number.
    """
    if episode not in actions_log:
        actions_log[episode] = {}
    for agent_name, action in actions.items():
        if agent_name not in actions_log[episode]:
            actions_log[episode][agent_name] = []
        actions_log[episode][agent_name].append(action.name)


def save_actions_log(actions_log, file_path="data/final_episode_log.json"):
    """
    Saves the action log to a JSON file.

    Parameters:
      - actions_log: Dictionary containing the logs for each episode.
      - file_path: Path where the log will be saved.
    """
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    with open(file_path, "w") as f:
        json.dump(actions_log, f, indent=4)


def update_successes(env, rewards_agents, successi_per_agente, done):
    """
    Updates the success count for agents who reached their goals.

    Parameters:
      - env: The environment containing agents.
      - rewards_agents: Dictionary of rewards for each agent.
      - successi_per_agente: Dictionary of success counts for each agent.
      - done: Dictionary indicating whether each agent has finished.
    """

    for agent in env.agents:
        rm_current_state = agent.reward_machine.get_current_state()
        rm_final_state = agent.reward_machine.get_final_state()
        if (
            done[agent.name]
            and rm_current_state == rm_final_state
            and rewards_agents[agent.name] > 0
        ):
            successi_per_agente[agent.name] += 1


def prepare_log_data(
    env,
    episode,
    rewards_agents,
    successi_per_agente,
    ricompense_per_episodio,
    finestra_media_mobile,
):
    """
    Prepares data for logging during training, including rewards and success rates.

    Parameters:
      - env: The environment containing agents.
      - episode: The current episode number.
      - rewards_agents: Dictionary of cumulative rewards for each agent.
      - successi_per_agente: Dictionary of success counts for each agent.
      - ricompense_per_episodio: List of rewards for each episode.
      - finestra_media_mobile: Window size for calculating moving averages.

    Returns:
      - log_data: A dictionary containing the prepared log data.
    """
    log_data = {"epsilon": env.epsilon, "episode": episode, "total_step": env.timestep}

    for agent in env.agents:
        agent_name = agent.name
        """if env.active_agents[agent.name] == False:
            continue"""
        reward = rewards_agents[agent_name]
        steps = env.agent_steps[agent_name]
        success_rate = (successi_per_agente[agent_name] / (episode + 1)) * 100
        ricompense_per_episodio[agent_name].append(reward)

        if len(ricompense_per_episodio[agent_name]) >= finestra_media_mobile:
            media_mobile = calcola_media_mobile(
                ricompense_per_episodio[agent_name], finestra_media_mobile
            )
            log_data[f"media_mobile_{agent_name}"] = media_mobile[-1]

        log_data.update(
            {
                f"reward_{agent_name}": reward,
                f"step_{agent_name}": steps,
                f"success_rate_training_{agent_name}": success_rate,
            }
        )

    return log_data


def get_epsilon_summary(agents):
    """
    Creates a summary of epsilon values for all agents.

    Parameters:
      - agents: List of agents.

    Returns:
      - A string summarizing epsilon values for each agent.
    """
    epsilon_parts = []
    for agent in agents:
        try:
            epsilon_parts.append(
                f"{agent.name}: {agent.learning_algorithm.epsilon:.2f}"
            )
        except:
            epsilon_parts.append(
                f"{agent.name}: N/A"
            )  # Per algoritmi che non usano epsilon
    return ", ".join(epsilon_parts)


def value_iteration(S, A, T, L, rm, gamma):
    """
    Standard value iteration to compute optimal policies for the grid environments.

    PARAMS
    ----------
    S:     List of states
    A:     List of actions
    T:     Transitions (it is a dictionary from SxA -> S)
    L:     Labeling function (it is a dictionary from states to events)
    rm:    Reward machine
    gamma: Discount factor

    RETURNS
    ----------
    Optimal deterministic policy (dictionary mapping from states (SxU) to actions)
    """
    U = rm.get_all_states()  # All states in the Reward Machine
    V = dict([((s, u), 0) for s in S for u in U])
    V_error = 1

    while V_error > 0.0000001:
        V_error = 0
        for s1 in S:
            for u1 in U:
                q_values = []
                for a in A:
                    s2 = T[(s1, a)]
                    l = "" if s2 not in L else L[s2]
                    u2, r = rm.get_reward_for_non_current_state(u1, l)

                    # Ensure correct transition logic
                    if u2 is None:  # If no transition, continue with the same state
                        u2 = u1
                    if u2 == rm.get_final_state():
                        done = True
                    else:
                        done = False

                    if done:
                        q_values.append(r)
                    else:
                        q_values.append(r + gamma * V[(s2, u2)])

                v_new = max(q_values)
                V_error = max([V_error, abs(v_new - V[(s1, u1)])])
                V[(s1, u1)] = v_new

    # Extracting the optimal policy
    policy = {}
    for s1 in S:
        for u1 in U:
            q_values = []
            for a in A:
                s2 = T[(s1, a)]
                l = "" if s2 not in L else L[s2]
                u2, r = rm.get_reward_for_non_current_state(u1, l)
                if u2 is None:
                    u2 = u1
                if u2 == rm.get_final_state():
                    done = True
                else:
                    done = False

                if done:
                    q_values.append(r)
                else:
                    q_values.append(r + gamma * V[(s2, u2)])

            a_i = max((x, i) for i, x in enumerate(q_values))[
                1
            ]  # argmax over the q-values
            policy[(s1, u1)] = A[a_i]

    return policy


def compute_normalized_arps(gamma, optimal_steps, avg_reward=1.0, target_arps=1.0):
    """
    Computes normalized ARPS so the final value equals the target ARPS.

    Parameters:
      - gamma: Discount factor.
      - optimal_steps: Number of optimal steps.
      - avg_reward: Average reward per step.
      - target_arps: Desired normalized ARPS value.

    Returns:
      - Normalized ARPS value.
    """
    if gamma == 1:
        # Handle gamma == 1 to avoid division by zero
        return target_arps / optimal_steps

    # Compute ARPS without normalization
    total_reward = avg_reward * (1 - gamma ** (optimal_steps + 1)) / (1 - gamma)
    arps = total_reward / optimal_steps

    # Compute required scaling factor
    scale_factor = target_arps / arps

    # Apply normalization
    normalized_total_reward = total_reward * scale_factor
    normalized_arps = normalized_total_reward / optimal_steps
    return normalized_arps


# -------------------------------------------------------------------------------------------
# Generic methods for comparing RL with VI
# -------------------------------------------------------------------------------------------


def extract_policy_from_qtable(agent):
    """
    Extracts the optimal (greedy) policy from the agent's Q-structure,
    whether it uses QLearning or QRMAX.

    For QLearning:
      - The algorithm has an attribute `.q_table` of shape (num_states, num_actions).
    For QRMAX (or similar):
      - The algorithm has an attribute `.Q` of shape (num_env_states, rm_states_count, num_actions).

    Returns:
      - policy_rl: 1D array of size [num_total_states],
                   where policy_rl[s] = best_action for state s.
    """
    alg = agent.get_learning_algorithm()

    # Case 1: QLearning, with .q_table
    if hasattr(alg, "q_table"):
        # q_table = (num_states, num_actions)
        q_table = alg.q_table
        num_states, num_actions = q_table.shape

        # Policy is simply the argmax across each row
        policy_rl = np.argmax(q_table, axis=1)  # shape (num_states,)

    # Case 2: QRMAX, with .Q
    elif hasattr(alg, "Q"):
        # Q = (num_env_states, rm_states_count, num_actions)
        q_table = alg.Q
        num_env_states, rm_states_count, num_actions = q_table.shape

        # We build the policy as an array of length (num_env_states * rm_states_count)
        num_states = num_env_states * rm_states_count
        policy_rl = np.zeros(num_states, dtype=int)

        for s in range(num_states):
            # Decompose s into (env_state, rm_state)
            rm_state = s % rm_states_count
            env_state = s // rm_states_count

            q_values = q_table[env_state, rm_state, :]
            best_action = np.argmax(q_values)
            policy_rl[s] = best_action

    else:
        raise ValueError(f"No 'q_table' or 'Q' attributes present in {alg}.")

    return policy_rl


def test_policy_opt_multi(
    rm_env,
    policy_dict,
    episodes_test=100,
    window_size=100,
    optimal_steps=30,
    gamma=0.9,
    test_deterministic=None,
):
    """
    Tests a multi-agent policy on an RMEnvironmentWrapper.

    Each agent has its own policy:
      - policy_dict[agent.name][encoded_state] = action (as an index).

    Parameters:
      - rm_env: RMEnvironmentWrapper or testing environment (will be copied).
      - policy_dict: Dictionary {agent.name -> array policy}, where
                     policy[s] = best_action_idx for that agent's state.
      - episodes_test: Number of testing episodes.
      - window_size: Moving average window size for success rate.
      - optimal_steps: Optimal number of steps (used for ARPS calculation).
      - gamma: Discount factor for discounted reward computation.
      - test_deterministic: If True, forces the environment into deterministic mode.

    Returns:
      - success_rate_per_agent: Success rate percentage for each agent.
      - moving_averages: Moving averages of success rate.
      - avg_timesteps_per_agent: Average timesteps per agent.
      - std_timesteps_per_agent: Standard deviation of timesteps per agent.
      - avg_reward_per_agent: Average reward per agent.
      - std_reward_per_agent: Standard deviation of rewards per agent.
      - avg_arps_per_agent: Average ARPS per agent.
      - rewards_per_agent: Rewards for potential statistical testing (e.g., t-tests).
    """

    # Copy the environment to not alter the original one
    env_test = copy.deepcopy(rm_env)

    # If we want to test deterministically, we set env_test.env.stochastic = False
    if test_deterministic is not None:
        try:
            env_test.env.stochastic = not test_deterministic
            if test_deterministic:
                episodes_test = 1
        except:
            print("Environment does not support setting stochastic mode")

    # Initialize counters and statistics for each agent
    successi_per_agente = {ag.name: 0 for ag in env_test.agents}
    successi_per_episodio = {ag.name: [] for ag in env_test.agents}
    ricompense_per_agente = {ag.name: [] for ag in env_test.agents}
    timestep_per_episodio = {ag.name: [] for ag in env_test.agents}
    arps_per_agente = {ag.name: [] for ag in env_test.agents}

    for episodio in range(episodes_test):
        # Reset environment for each episode
        states, infos = env_test.reset(10000 + episodio)

        done = {ag.name: False for ag in env_test.agents}
        agent_success = {ag.name: False for ag in env_test.agents}
        timestep = 0
        # Reward discounted accumulated in this episode for each agent
        episode_rewards = {ag.name: 0 for ag in env_test.agents}
        cum_gamma = 1.0

        # Until all agents are "done" or we pass 1000 steps
        while not all(done.values()) and timestep < 1000:
            actions = {}
            for ag in env_test.agents:
                # Agent current state
                current_state = env_test.env.get_state(ag)
                # Encode the state with the agent encoder
                encoded_state, _ = ag.encoder.encode(current_state)

                # Gets the ag-specific multi-agent policy
                # Example: policy_ag = policy_dict[ag.name]
                # The index action = policy_ag[encoded_state]
                if ag.name not in policy_dict:
                    raise ValueError(
                        f"No policy provided for agent {ag.name} in policy_dict."
                    )
                policy_ag = policy_dict[ag.name]

                # Indice azione
                if encoded_state >= len(policy_ag):
                    raise ValueError(
                        f"encoded_state={encoded_state} outside policy_ag length {len(policy_ag)}"
                    )
                action_index = policy_ag[encoded_state]

                # Get the corresponding ActionRL
                all_actions = ag.get_actions()  # list of possible actions
                if action_index < 0 or action_index >= len(all_actions):
                    raise ValueError(
                        f"Action index {action_index} invalid for state {encoded_state} (agent {ag.name})."
                    )
                action = all_actions[action_index]
                actions[ag.name] = action

            # Perform the step on ALL agents
            new_states, rewards, done, truncations, infos = env_test.step(actions)

            # Update rewards (discounted)
            for ag in env_test.agents:
                if not agent_success[ag.name]:
                    episode_rewards[ag.name] += cum_gamma * rewards[ag.name]
                    # Check whether the agent reached the final Reward Machine state
                    if (
                        done[ag.name]
                        and ag.get_reward_machine().get_current_state()
                        == ag.get_reward_machine().get_final_state()
                    ):
                        successi_per_agente[ag.name] += 1
                        agent_success[ag.name] = True

            # Next step
            states = copy.deepcopy(new_states)
            cum_gamma *= gamma
            timestep += 1

        # end of an episode
        for ag in env_test.agents:
            # Record success and outcomes
            successi_per_episodio[ag.name].append(1 if agent_success[ag.name] else 0)
            # Let's save the final reward accumulated in the episode
            ricompense_per_agente[ag.name].append(episode_rewards[ag.name])
            if agent_success[ag.name]:
                timestep_per_episodio[ag.name].append(timestep)

            # Compute discounted ARPS
            if timestep > 0:
                arps = (episode_rewards[ag.name] / timestep) / optimal_steps
                arps_per_agente[ag.name].append(arps)

    # End of all episodes

    # Calculation of success rate, moving average, timesteps, reward...
    success_rate_per_agente = {
        ag.name: (successi_per_agente[ag.name] / episodes_test) * 100
        for ag in env_test.agents
    }

    # If you want to save the moving averages of the success rate
    moving_averages = {}
    for ag in env_test.agents:
        succ_array = successi_per_episodio[ag.name]
        if len(succ_array) >= window_size:
            moving_averages[ag.name] = (
                np.convolve(succ_array, np.ones(window_size), "valid") / window_size
            )
        else:
            moving_averages[ag.name] = np.array([])

    avg_timesteps_per_agente = {}
    std_timesteps_per_agente = {}
    for ag in env_test.agents:
        if len(timestep_per_episodio[ag.name]) > 0:
            avg_timesteps_per_agente[ag.name] = np.mean(timestep_per_episodio[ag.name])
            std_timesteps_per_agente[ag.name] = np.std(timestep_per_episodio[ag.name])
        else:
            avg_timesteps_per_agente[ag.name] = 0
            std_timesteps_per_agente[ag.name] = 0

    avg_reward_per_agente = {}
    std_reward_per_agente = {}
    for ag in env_test.agents:
        all_rews = ricompense_per_agente[ag.name]
        if len(all_rews) > 0:
            avg_reward_per_agente[ag.name] = np.mean(all_rews)
            std_reward_per_agente[ag.name] = np.std(all_rews)
        else:
            avg_reward_per_agente[ag.name] = 0
            std_reward_per_agente[ag.name] = 0

    avg_arps_per_agente = {}
    for ag in env_test.agents:
        all_arps = arps_per_agente[ag.name]
        if len(all_arps) > 0:
            avg_arps_per_agente[ag.name] = np.mean(all_arps)
        else:
            avg_arps_per_agente[ag.name] = 0

    return (
        success_rate_per_agente,
        moving_averages,
        avg_timesteps_per_agente,
        std_timesteps_per_agente,
        avg_reward_per_agente,
        std_reward_per_agente,
        avg_arps_per_agente,
        ricompense_per_agente,  # useful for t-test
    )


def perform_ttest(rewards_policy1, rewards_policy2, agente_name):
    """
    Performs a t-test to compare the rewards from two different policies for a given agent.

    Parameters:
      - rewards_policy1: Dictionary {agent_name: [list_of_rewards]} for the first policy.
      - rewards_policy2: Dictionary {agent_name: [list_of_rewards]} for the second policy.
      - agente_name: Name of the agent being evaluated.
    Returns:
      - statistic: The t-statistic value.
      - pvalue: The p-value from the t-test.
    """
    # Extract reward lists for the specified agent
    r1 = rewards_policy1[agente_name]
    r2 = rewards_policy2[agente_name]

    statistic, pvalue = ttest_ind(r1, r2, equal_var=False)
    return statistic, pvalue


def estrai_policy_rl(agent):
    """
    Extracts the policy from an RL agent by selecting the action with the maximum
    Q-value for each state.

    Parameters:
      - agent: An instance of an RL agent.
    Returns:
      - policy_rl: 1D array where each element is the index of the best action for the given state.
    """
    algo = agent.get_learning_algorithm()
    if hasattr(algo, "Q"):
        q_table = algo.Q
    elif hasattr(algo, "q_table"):
        q_table = algo.q_table
    else:
        raise ValueError("The RL algorithm does not expose an accessible Q-table.")

    policy_rl = np.argmax(q_table, axis=1)  # policy_rl[s] = argmax_a Q(s,a)
    return policy_rl


def extract_value_function_from_qtable(q_table):
    """
    Extracts the value function V(s) from a Q-table by selecting the maximum Q-value
    for each state.

    Parameters:
      - q_table: NumPy array of shape (num_env_states, rm_states_count, num_actions)
                 or (num_states, num_actions).
    Returns:
      - value_function: 1D NumPy array where value_function[s] is the maximum Q-value for state s.
    """
    # If the Q-table is two-dimensional
    if len(q_table.shape) == 2:
        num_states, num_actions = q_table.shape
        value_function = np.max(q_table, axis=1)  # Max Q-value per state
    # If the Q-table is three-dimensional
    elif len(q_table.shape) == 3:
        num_env_states, rm_states_count, num_actions = q_table.shape
        num_states = num_env_states * rm_states_count
        value_function = np.zeros(num_states)
        for s in range(num_states):
            rm_state = s % rm_states_count
            env_state = s // rm_states_count
            q_values = q_table[env_state, rm_state, :]  # Q-values for all actions
            value_function[s] = np.max(q_values)  # Value function is max Q-value
    else:
        raise ValueError("Unexpected Q-table shape. Expected 2D or 3D array.")

    return value_function


def extract_value_function_from_policy(policy_rl, q_table):
    """
    Extracts the value function V(s) for a given policy and Q-table.

    Parameters:
      - policy_rl: 1D array where policy_rl[s] is the action index for state s.
      - q_table: NumPy array of shape (num_env_states, rm_states_count, num_actions)
                 or (num_states, num_actions).
    Returns:
      - value_function: 1D NumPy array where value_function[s] is V(s) for state s.
    """
    if not isinstance(q_table, np.ndarray):
        raise ValueError("Q-table is not a NumPy array. Received type:", type(q_table))

    print("Q-table shape:", q_table.shape)  # Debugging

    if len(q_table.shape) == 2:  # Q-table is (num_states, num_actions)
        num_states, num_actions = q_table.shape
        value_function = np.zeros(num_states)
        for s in range(num_states):
            action = policy_rl[s]
            value_function[s] = q_table[s, action]
    elif (
        len(q_table.shape) == 3
    ):  # Q-table is (num_env_states, rm_states_count, num_actions)
        num_env_states, rm_states_count, num_actions = q_table.shape
        num_states = num_env_states * rm_states_count
        value_function = np.zeros(num_states)
        for s in range(num_states):
            rm_state = s % rm_states_count
            env_state = s // rm_states_count
            action = policy_rl[s]
            value_function[s] = q_table[env_state, rm_state, action]
    else:
        raise ValueError(f"Unexpected Q-table shape: {q_table.shape}")

    return value_function
