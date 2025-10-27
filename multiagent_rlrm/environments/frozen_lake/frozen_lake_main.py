import numpy as np
from multiagent_rlrm.multi_agent.reward_machine import RewardMachine
from multiagent_rlrm.multi_agent.agent_rl import AgentRL
from multiagent_rlrm.multi_agent.action_rl import ActionRL
from multiagent_rlrm.learning_algorithms.qlearning import QLearning
from multiagent_rlrm.learning_algorithms.rmax import RMax
from multiagent_rlrm.learning_algorithms.qrmax_v2 import QRMax
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
)  # Import del nuovo EventDetector
from multiagent_rlrm.multi_agent.wrappers.rm_environment_wrapper import (
    RMEnvironmentWrapper,
)  # Import del wrapper


NUM_EPISODES = 20001
grid_height = 10
grid_width = 10
WANDB_PROJECT = "ma_frozen_lake"
WANDB_ENTITY = "..."


# Initialize WandB
# wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, mode="disabled")
wandb.init(project="maze_RL", entity="alee8")  # , mode="disabled")
# Environment Setup
map_frozenk_lake = """
  2 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩  5
 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩
 🟩 🟩 🟩 ⛔ ⛔ 🟩 🟩 🟩 🟩 🟩
 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩
 🟩 🟩 🟩 🟩  1 🟩 🟩 🟩 🟩 🟩
 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩
 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩
 ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ ⛔ 🟩 ⛔ ⛔
 🟩 🟩 🟩 🟩  3 🟩 🟩 🟩 🟩 🟩
  4 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩 🟩
 """

holes, goals = parse_map_emoji(map_frozenk_lake)
print("H: ", holes, "G: ", goals)
object_positions = {
    "holes": holes,
}

env = MultiAgentFrozenLake(
    width=grid_width,
    height=grid_height,
    holes=holes,
)


env.frozen_lake = True
env.penalty_amount = 0
env.delay_action = False  # Abilita la funzione "wait"


a3 = AgentRL("a3", env)
a1 = AgentRL("a1", env)
a1.set_initial_position(6, 0)  # Aggiungo la pos anche allo stato dell'agente
a3.set_initial_position(4, 0)  # Aggiungo la pos anche allo stato dell'agente


a3.add_state_encoder(StateEncoderFrozenLake(a3))
a1.add_state_encoder(StateEncoderFrozenLake(a1))
# Definizione delle precondizioni
def can_move_up(agent):
    return agent.get_position()[1] > 0


def can_move_down(agent):
    return agent.get_position()[1] < env.grid_height - 1


def can_move_left(agent):
    return agent.get_position()[0] > 0


def can_move_right(agent):
    return agent.get_position()[0] < env.grid_width - 1


# Definizione degli effetti intenzionali (gestiti poi stocasticamente dall'ambiente)
def effect_up(agent):
    agent.set_position(agent.get_position()[0], agent.get_position()[1] - 1)


def effect_down(agent):
    agent.set_position(agent.get_position()[0], agent.get_position()[1] + 1)


def effect_left(agent):
    agent.set_position(agent.get_position()[0] - 1, agent.get_position()[1])


def effect_right(agent):
    agent.set_position(agent.get_position()[0] + 1, agent.get_position()[1])


# Creazione delle azioni
move_up = ActionRL("up", [can_move_up], [effect_up])
move_down = ActionRL("down", [can_move_down], [effect_down])
move_left = ActionRL("left", [can_move_left], [effect_left])
move_right = ActionRL("right", [can_move_right], [effect_right])

# Aggiunta delle azioni agli agenti
a1.add_action(move_up)
a1.add_action(move_down)
a1.add_action(move_left)
a1.add_action(move_right)

a3.add_action(move_up)
a3.add_action(move_down)
a3.add_action(move_left)
a3.add_action(move_right)

# Definisci gli eventi che rappresentano l'agente che raggiunge le celle specifiche
event_reach_cell_1 = (4, 4)
event_reach_cell_2 = (0, 0)
event_reach_cell_3 = (4, 8)
event_reach_cell_4 = (0, 9)
event_reach_cell_5 = (9, 0)

# Definisci gli stati della RM
state_start = "state0"
state_reached_1 = "state1"
state_reached_2 = "state2"
state_reached_3 = "state3"
state_reached_4 = "state4"
state_reached_5 = "state5"

# Definisci le transizioni della RM
# {(stato_corrente, evento): (nuovo_stato, ricompensa)}
transitions = {
    (state_start, event_reach_cell_1): (state_reached_1, 10),
    (state_reached_1, event_reach_cell_2): (state_reached_2, 15),
    (state_reached_2, event_reach_cell_3): (state_reached_3, 25),
    (state_reached_3, event_reach_cell_4): (state_reached_4, 30),
    (state_reached_4, event_reach_cell_5): (state_reached_5, 100),
}

# Creazione degli EventDetector
positions = {
    event_reach_cell_1,
    event_reach_cell_2,
    event_reach_cell_3,
    event_reach_cell_4,
    event_reach_cell_5,
}
event_detector = PositionEventDetector(positions)

# Crea la RM
RM_1 = RewardMachine(transitions, event_detector)
RM_3 = RewardMachine(transitions, event_detector)

a1.set_reward_machine(RM_1)
a3.set_reward_machine(RM_3)

# env.add_agent(a1)
env.add_agent(a3)

# Avvolgi l'ambiente con il wrapper RMEnvironmentWrapper
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
    learning_rate=0.2,
    gamma=0.99,
    action_selection="greedy",
    epsilon_start=0.01,
    epsilon_end=0.01,
    epsilon_decay=0.9995,
)

q_learning3 = QLearning(
    state_space_size=env.grid_width * env.grid_height * RM_3.numbers_state(),
    action_space_size=4,
    learning_rate=0.2,
    gamma=0.99,
    action_selection="greedy",
    epsilon_start=0.01,
    epsilon_end=0.01,
    epsilon_decay=0.9995,
    use_crm=False,
)

q_learning_lambda = QLearningLambda(
    state_space_size=env.grid_width
    * env.grid_height
    * RM_3.numbers_state(),  # env.num_rm_states,
    action_space_size=4,
    learning_rate=0.1,
    gamma=0.99,
    lambd=0.8,
    action_selection="softmax",
    epsilon_start=0.01,
    epsilon_end=0.01,
    epsilon_decay=0.9995,
)


qrmax1 = QRMax(
    state_space_size=env.grid_width * env.grid_height * RM_1.numbers_state(),
    action_space_size=4,
    gamma=0.99,
    q_space_size=4,
    nsamplesTE=100,  # Transition Environment - soglia per considerare nota una transizione (s, a) nell'ambiente
    nsamplesRE=1,  # Reward Environment - soglia per considerare nota la ricompensa associata a una coppia (s, a) nell'ambiente
    nsamplesTQ=1,  # Transition for Q - soglia per considerare nota una transizione di stato dell'automa di Reward Machine (q, s') data una coppia (s, a)
    nsamplesRQ=1,  # Reward for Q - soglia per considerare nota la ricompensa associata a una transizione dell'automa di Reward Machine (q, s', q')
    # seed=args.seed,
)

qrmax3 = QRMax(
    state_space_size=env.grid_width * env.grid_height * RM_3.numbers_state(),
    action_space_size=4,
    gamma=0.99,
    q_space_size=4,
    nsamplesTE=100,  # Transition Environment - soglia per considerare nota una transizione (s, a) nell'ambiente
    nsamplesRE=1,  # Reward Environment - soglia per considerare nota la ricompensa associata a una coppia (s, a) nell'ambiente
    nsamplesTQ=1,  # Transition for Q - soglia per considerare nota una transizione di stato dell'automa di Reward Machine (q, s') data una coppia (s, a)
    nsamplesRQ=1,  # Reward for Q - soglia per considerare nota la ricompensa associata a una transizione dell'automa di Reward Machine (q, s', q')
    # seed=args.seed,
)


# a1.set_learning_algorithm(qrmax1)
a3.set_learning_algorithm(q_learning3)

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
rm_env.reset()
# a1.get_learning_algorithm().learn_init()
a3.get_learning_algorithm().learn_init()
from multiagent_rlrm.utils.utils import *

for episode in range(NUM_EPISODES):
    states, infos = rm_env.reset()
    states = copy.deepcopy(states)
    done = {a.name: False for a in rm_env.agents}
    rewards_agents = {
        a.name: 0 for a in rm_env.agents
    }  # Inizializza le ricompense episodiche
    record_episode = episode % 10000 == 0 and episode != 0
    # record_episode = False
    if record_episode:
        renderer.render(episode, states)  # Cattura frame durante l'episodio

    while not all(done.values()):
        # states, infos = rm_env.reset()
        # states = copy.deepcopy(states)
        actions = {}
        rewards = {a.name: 0 for a in rm_env.agents}
        # infos = {a.name: {} for a in rm_env.agents}
        for ag in rm_env.agents:
            current_state = rm_env.env.get_state(ag)
            action = ag.select_action(current_state)
            actions[ag.name] = action
            # Log delle azioni nell'ultimo episodio
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
        # end-training

        if record_episode:
            renderer.render(episode, states)  # Cattura frame durante l'episodio

        if all(truncations.values() or done.values()):
            break

    if record_episode:
        renderer.save_episode(episode)  # Salva il video solo alla fine dell'episodio

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
        success_rate_per_agente = test_policy_optima(rm_env, episodi_test=100)
        for ag_name, success_rate in success_rate_per_agente.items():
            log_data[f"success_rate_optima_{ag_name}"] = success_rate"""

    wandb.log(log_data, step=episode)
    epsilon_str = get_epsilon_summary(rm_env.agents)

    print(
        f"Episodio {episode + 1}: Ricompensa = {rewards_agents}, Total Step: {rm_env.env.timestep}, Agents Step = {rm_env.env.agent_steps}, Epsilon agents= [{epsilon_str}]"
    )


# Salva la Q-table all'ultimo episodio
save_q_tables(rm_env.agents)
# After training or during training at specified intervals
data = np.load("data/q_tables.npz")
generate_heatmaps_for_agents(rm_env.agents, data, grid_dims=(grid_width, grid_height))
