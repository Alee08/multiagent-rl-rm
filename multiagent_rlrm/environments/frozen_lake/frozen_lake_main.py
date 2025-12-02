import numpy as np
from multiagent_rlrm.multi_agent.reward_machine import RewardMachine
from multiagent_rlrm.multi_agent.agent_rl import AgentRL
from multiagent_rlrm.multi_agent.action_encoder import ActionEncoder
from multiagent_rlrm.learning_algorithms.qlearning import QLearning
from multiagent_rlrm.learning_algorithms.rmax import RMax
from multiagent_rlrm.learning_algorithms.qrmax_v2 import QRMax_v2
from multiagent_rlrm.learning_algorithms.qlearning_lambda import QLearningLambda
from multiagent_rlrm.utils.utils import encode_state, parse_map_string, parse_map_emoji
from multiagent_rlrm.render.render import EnvironmentRenderer
from multiagent_rlrm.environments.frozen_lake.state_encoder_frozen_lake import (
    StateEncoderFrozenLake,
)
from multiagent_rlrm.environments.frozen_lake.ma_frozen_lake import (
    MultiAgentFrozenLake,
)
from multiagent_rlrm.render.heatmap import (
    generate_heatmaps,
    generate_heatmaps_for_agents,
)
from multiagent_rlrm.environments.utils_envs.evaluation_metrics import *
import wandb
import copy
import json
import os
from multiagent_rlrm.environments.frozen_lake.detect_event import (
    PositionEventDetector,
)  # Import the new EventDetector
from multiagent_rlrm.multi_agent.wrappers.rm_environment_wrapper import (
    RMEnvironmentWrapper,
)  # Import the wrapper
from multiagent_rlrm.environments.frozen_lake.action_encoder_frozen_lake import (
    ActionEncoderFrozenLake,
)


NUM_EPISODES = 30000
# grid_height = 10
# grid_width = 10
WANDB_PROJECT = "ma_frozen_lake"
WANDB_ENTITY = "..."


# Initialize WandB
# wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, mode="disabled")
wandb.init(project="deep_FL", entity="alee8", mode="disabled")

map_frozenk_lake10x10 = """
  B 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩
 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩
 🟩 🟩 🟩 ⛔ ⛔ 🟩 🟩 🟩 🟩 🟩
 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩
 🟩 🟩 🟩 🟩 A  🟩 🟩 🟩 🟩 🟩
 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩
 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩
 ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ 🟩 ⛔ ⛔
 🟩 🟩 🟩 🟩  C 🟩 🟩 🟩 🟩 🟩
 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩
 """

map_frozenk_lake5x5 = """
 B  🟩 🟩 🟩 🟩 
 🟩 🟩 🟩 🟩 🟩
 🟩 🟩 🟩 ⛔ ⛔
 🟩 🟩 🟩 🟩 🟩
 ⛔ ⛔ ⛔ ⛔ A
 """

map_frozenk_lake3x3 = """
 🟩 🟩 🟩 
 🟩 ⛔ 🟩 
 🟩 🟩  A 
 """

map = map_frozenk_lake10x10
holes, goals, dimensions = parse_map_emoji(map)
# holes, goals = parse_map_emoji(map_frozenk_lake3x3)
print("Holes:", holes)
print("Goals:", goals)
print("Dimensions:", dimensions)
object_positions = {
    "holes": holes,
}
env = MultiAgentFrozenLake(
    width=dimensions[0],
    height=dimensions[1],
    holes=holes,
)
# breakpoint()

env.frozen_lake_stochastic = False
env.penalty_amount = 0
env.delay_action = False  # Enable delayed "wait" action behavior


a3 = AgentRL("a3", env)
a1 = AgentRL("a1", env)
a1.set_initial_position(0, 0)  # Also add position to the agent's state
a3.set_initial_position(1, 0)  # Also add position to the agent's state


a3.add_state_encoder(StateEncoderFrozenLake(a3))
a1.add_state_encoder(StateEncoderFrozenLake(a1))

a1.add_action_encoder(ActionEncoderFrozenLake(a1))
a3.add_action_encoder(ActionEncoderFrozenLake(a3))


# Define Reward Machine transitions
# {(current_state, event): (next_state, reward)}
transitions = {
    ("state0", goals["A"]): ("state1", 10),
    ("state1", goals["B"]): ("state2", 15),
    ("state2", goals["C"]): ("state3", 20),
}

# Create EventDetectors
positions = {goals["A"], goals["B"], goals["C"]}
event_detector = PositionEventDetector(positions)

# Create Reward Machines
RM_1 = RewardMachine(transitions, event_detector)
RM_3 = RewardMachine(transitions, event_detector)

a1.set_reward_machine(RM_1)
a3.set_reward_machine(RM_3)

# env.add_agent(a1)
env.add_agent(a3)

# Wrap the environment with RMEnvironmentWrapper
rm_env = RMEnvironmentWrapper(env, [a3])

rmax = RMax(
    state_space_size=env.grid_width * env.grid_height * RM_3.numbers_state(),
    action_space_size=4,
    s_a_threshold=100,
    max_reward=1,
    gamma=0.99,
    epsilon_one=0.99,
)

q_learning1 = QLearning(
    state_space_size=env.grid_width * env.grid_height * RM_1.numbers_state(),
    action_space_size=4,
    learning_rate=1,
    gamma=0.99,
    action_selection="greedy",
    epsilon_start=0.01,
    epsilon_end=0.01,
    epsilon_decay=0.9995,
)

q_learning3 = QLearning(
    state_space_size=env.grid_width * env.grid_height * RM_3.numbers_state(),
    action_space_size=4,
    learning_rate=1,
    gamma=0.99,
    action_selection="greedy",
    epsilon_start=0.01,
    epsilon_end=0.01,
    epsilon_decay=0.9995,
    qtable_init=2,
    use_qrm=True,
)


qrmax1 = QRMax_v2(
    state_space_size=env.grid_width * env.grid_height * RM_1.numbers_state(),
    action_space_size=4,
    gamma=0.99,
    q_space_size=4,
    nsamplesTE=100,  # Transition Environment - threshold to consider (s, a) transition in the environment as known
    nsamplesRE=1,  # Reward Environment - threshold to consider the reward associated with a pair (s, a) as known
    nsamplesTQ=1,  # Transition for Q - threshold to consider a RM state transition (q, s') given (s, a) as known
    nsamplesRQ=1,  # Reward for Q - threshold to consider the reward of a RM transition (q, s', q') as known
    # seed=args.seed,
)

qrmax3 = QRMax_v2(
    state_space_size=env.grid_width * env.grid_height * RM_3.numbers_state(),
    action_space_size=4,
    gamma=0.99,
    q_space_size=4,
    nsamplesTE=100,  # Transition Environment - threshold to consider (s, a) transition in the environment as known
    nsamplesRE=1,  # Reward Environment - threshold to consider the reward associated with a pair (s, a) as known
    nsamplesTQ=1,  # Transition for Q - threshold to consider a RM state transition (q, s') given (s, a) as known
    nsamplesRQ=1,  # Reward for Q - threshold to consider the reward of a RM transition (q, s', q') as known
    # seed=args.seed,
)


# a1.set_learning_algorithm(qrmax1)
a3.set_learning_algorithm(q_learning3)

# Load QTABLE
# a3.get_learning_algorithm().load_qtable("q_table.pkl")

renderer = EnvironmentRenderer(
    env.grid_width,
    env.grid_height,
    agents=env.agents,
    object_positions=object_positions,
    goals=goals,
)

renderer.init_pygame()


successi_per_agente = {agent.name: 0 for agent in rm_env.agents}
ricompense_per_episodio = {agent.name: [] for agent in rm_env.agents}
finestra_media_mobile = 1000
actions_log = {agent.name: [] for agent in rm_env.agents}
success_counts = {agent.name: 0 for agent in rm_env.agents}
q_tables = {}
rm_env.reset(111)
# a1.get_learning_algorithm().learn_init()
a3.get_learning_algorithm().learn_init()
from multiagent_rlrm.utils.utils import *

for episode in range(NUM_EPISODES):
    states, infos = rm_env.reset(111)
    done = {a.name: False for a in rm_env.agents}
    rewards_agents = {
        a.name: 0 for a in rm_env.agents
    }  # Initialize per-episode rewards
    record_episode = episode % 200000 == 0 and episode != 0
    # record_episode = False
    if record_episode:
        renderer.render(episode, states)  # Capture frames during the episode

    while not all(done.values()):
        actions = {}
        rewards = {a.name: 0 for a in rm_env.agents}
        infos = {a.name: {} for a in rm_env.agents}
        for ag in rm_env.agents:
            current_state = rm_env.env.get_state(ag)
            action = ag.select_action(current_state)
            actions[ag.name] = action
            # Log actions in the last episode
            update_actions_log(actions_log, actions, NUM_EPISODES)

        new_states, rewards, done, truncations, infos = rm_env.step(actions)

        for agent in rm_env.agents:
            """if not rm_env.env.active_agents[agent.name]:
            continue"""
            terminated = done[agent.name] or truncations[agent.name]
            agent.update_policy(
                state=states[agent.name],
                action=actions[agent.name],
                reward=rewards[agent.name],
                next_state=new_states[agent.name],
                terminated=terminated,
                infos=infos[agent.name],
            )

            rewards_agents[agent.name] += rewards[agent.name]
        states = copy.deepcopy(new_states)
        # end-training step

        if record_episode:
            renderer.render(episode, states)  # Capture frames during the episode

        if all(truncations.values()):
            break
    if record_episode:
        renderer.save_episode(episode)  # Save video only at the end of the episode

    update_successes(rm_env.env, rewards_agents, successi_per_agente, done)
    log_data = prepare_log_data(
        rm_env.env,
        episode,
        rewards_agents,
        successi_per_agente,
        ricompense_per_episodio,
        finestra_media_mobile,
    )

    """if episode % 1000 == 0:
        # Evaluate policy every 1000 episodes
        success_rate_per_agente = test_policy_optima(rm_env, episodi_test=100)
        for ag_name, success_rate in success_rate_per_agente.items():
            log_data[f"success_rate_optima_{ag_name}"] = success_rate"""

    wandb.log(log_data, step=episode)
    epsilon_str = get_epsilon_summary(rm_env.agents)

    print(
        f"Episodio {episode + 1}: Ricompensa = {rewards_agents}, Total Step: {rm_env.env.timestep}, Agents Step = {rm_env.env.agent_steps}, Epsilon agents= [{epsilon_str}]"
    )

# Save QTABLE
a3.get_learning_algorithm().save_qtable("q_table_100x100.pkl")
# Save Q-tables at the last episode
save_q_tables(rm_env.agents)
# After training or during training at specified intervals
data = np.load(f"data/q_tables.npz")
generate_heatmaps_for_agents(
    rm_env.agents, data, grid_dims=(dimensions[0], dimensions[1])
)
