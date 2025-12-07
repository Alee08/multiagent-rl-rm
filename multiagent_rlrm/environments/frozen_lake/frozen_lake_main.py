import copy
import numpy as np
import wandb

from multiagent_rlrm.environments.frozen_lake.action_encoder_frozen_lake import (
    ActionEncoderFrozenLake,
)
from multiagent_rlrm.environments.frozen_lake.detect_event import PositionEventDetector
from multiagent_rlrm.environments.frozen_lake.ma_frozen_lake import (
    MultiAgentFrozenLake,
)
from multiagent_rlrm.environments.frozen_lake.state_encoder_frozen_lake import (
    StateEncoderFrozenLake,
)
from multiagent_rlrm.environments.utils_envs.evaluation_metrics import (
    get_epsilon_summary,
    prepare_log_data,
    save_q_tables,
    update_actions_log,
    update_successes,
)
from multiagent_rlrm.learning_algorithms.qlearning import QLearning
from multiagent_rlrm.learning_algorithms.qrmax_v2 import QRMax_v2
from multiagent_rlrm.multi_agent.agent_rl import AgentRL
from multiagent_rlrm.multi_agent.reward_machine import RewardMachine
from multiagent_rlrm.multi_agent.wrappers.rm_environment_wrapper import (
    RMEnvironmentWrapper,
)
from multiagent_rlrm.render.heatmap import generate_heatmaps_for_agents
from multiagent_rlrm.render.render import EnvironmentRenderer
from multiagent_rlrm.utils.utils import parse_map_emoji


NUM_EPISODES = 30_000
RENDER_EVERY = 100  # Set to None to disable video rendering
SEED = 111

# WandB is disabled; keep init for a consistent logging interface
wandb.init(project="deep_FL", entity="alee8", mode="disabled")

# Map with three goals (A, B, C) and some holes
MAP_LAYOUT = """
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

# --- Environment and agents ----------------------------------------------- #
holes, goals, dimensions = parse_map_emoji(MAP_LAYOUT)
object_positions = {"holes": holes}

env = MultiAgentFrozenLake(
    width=dimensions[0],
    height=dimensions[1],
    holes=holes,
)
env.frozen_lake_stochastic = False
env.penalty_amount = 0
env.delay_action = False

# Two agents: a1 learns with Q-Learning, a2 with QRMax
a1 = AgentRL("a1", env)
a2 = AgentRL("a2", env)

a1.set_initial_position(5, 0)
a2.set_initial_position(0, 0)

for ag in (a1, a2):
    ag.add_state_encoder(StateEncoderFrozenLake(ag))
    ag.add_action_encoder(ActionEncoderFrozenLake(ag))

# Reward Machine: reach A -> B -> C
transitions = {
    ("state0", goals["A"]): ("state1", 10),
    ("state1", goals["B"]): ("state2", 15),
    ("state2", goals["C"]): ("state3", 20),
}
event_detector = PositionEventDetector(set(goals.values()))
rm_q = RewardMachine(transitions, event_detector)
rm_qr = RewardMachine(transitions, event_detector)

a1.set_reward_machine(rm_q)
a2.set_reward_machine(rm_qr)

env.add_agent(a1)
env.add_agent(a2)
rm_env = RMEnvironmentWrapper(env, [a1, a2])

# Tabular Q-Learning for a1
q_learning = QLearning(
    state_space_size=env.grid_width * env.grid_height * rm_q.numbers_state(),
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
a1.set_learning_algorithm(q_learning)

# Model-based QRMax for a2
q_learning2 = QLearning(
    state_space_size=env.grid_width * env.grid_height * rm_qr.numbers_state(),
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
a2.set_learning_algorithm(q_learning2)

renderer = EnvironmentRenderer(
    env.grid_width,
    env.grid_height,
    agents=env.agents,
    object_positions=object_positions,
    goals=goals,
)
renderer.init_pygame()

# --- Stats and logging ----------------------------------------------------- #
success_per_agent = {agent.name: 0 for agent in rm_env.agents}
rewards_per_episode = {agent.name: [] for agent in rm_env.agents}
moving_avg_window = 1000
actions_log = {}

rm_env.reset(SEED)
a1.get_learning_algorithm().learn_init()
a2.get_learning_algorithm().learn_init()

# Helper to run and record a greedy (best-action) episode
def record_greedy_episode(tag="greedy", seed=SEED):
    renderer.frames = []  # Clear previous recording frames
    states, infos = rm_env.reset(seed)
    done = {a.name: False for a in rm_env.agents}
    renderer.render(tag, states)

    while not all(done.values()):
        actions = {}
        for ag in rm_env.agents:
            current_state = rm_env.env.get_state(ag)
            actions[ag.name] = ag.select_action(current_state, best=True)

        states, rewards, done, truncations, infos = rm_env.step(actions)
        renderer.render(tag, states)

        if all(truncations.values()):
            break

    renderer.save_episode(tag)


# --- Training loop --------------------------------------------------------- #
for episode in range(NUM_EPISODES):
    states, infos = rm_env.reset(SEED)
    done = {a.name: False for a in rm_env.agents}
    rewards_agents = {a.name: 0 for a in rm_env.agents}
    record_episode = bool(RENDER_EVERY) and episode % RENDER_EVERY == 0

    if record_episode:
        renderer.render(episode, states)

    while not all(done.values()):
        actions = {}
        rewards = {a.name: 0 for a in rm_env.agents}
        infos = {a.name: {} for a in rm_env.agents}

        for ag in rm_env.agents:
            current_state = rm_env.env.get_state(ag)
            actions[ag.name] = ag.select_action(current_state)

        update_actions_log(actions_log, actions, episode)

        new_states, rewards, done, truncations, infos = rm_env.step(actions)

        for ag in rm_env.agents:
            terminated = done[ag.name] or truncations[ag.name]
            ag.update_policy(
                state=states[ag.name],
                action=actions[ag.name],
                reward=rewards[ag.name],
                next_state=new_states[ag.name],
                terminated=terminated,
                infos=infos[ag.name],
            )
            rewards_agents[ag.name] += rewards[ag.name]

        states = copy.deepcopy(new_states)

        if record_episode:
            renderer.render(episode, states)

        if all(truncations.values()):
            break

    if record_episode:
        renderer.save_episode(episode)

    update_successes(rm_env.env, rewards_agents, success_per_agent, done)
    log_data = prepare_log_data(
        rm_env.env,
        episode,
        rewards_agents,
        success_per_agent,
        rewards_per_episode,
        moving_avg_window,
    )

    wandb.log(log_data, step=episode)
    epsilon_str = get_epsilon_summary(rm_env.agents)
    print(
        f"Episode {episode + 1}: Rewards = {rewards_agents}, "
        f"Total Step: {rm_env.env.timestep}, Agents Step = {rm_env.env.agent_steps}, "
        f"Epsilon agents= [{epsilon_str}]"
    )

    # Optional greedy roll-out recording every RENDER_EVERY episodes
    if RENDER_EVERY and episode > 0 and episode % RENDER_EVERY == 0:
        print(f"Recording greedy episode at training episode {episode}...")
        record_greedy_episode(tag=f"greedy_ep_{episode}", seed=SEED)

# --- Save results ---------------------------------------------------------- #
save_q_tables(rm_env.agents)
data = np.load("data/q_tables.npz")
generate_heatmaps_for_agents(
    rm_env.agents, data, grid_dims=(dimensions[0], dimensions[1])
)
