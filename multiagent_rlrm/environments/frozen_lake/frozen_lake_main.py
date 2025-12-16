"""
Example: multi-agent FrozenLake with Reward Machines (RMs).

By default this script uses a small, hand-written RM (A -> B -> C) to show the
core library usage (env + agents + RM + training loop).

Optional (NL -> RM): generate an RM spec from natural language with `rmgen` and
load it here via `--rm-spec` (shared by both agents) or `--rm-spec-a1/--rm-spec-a2`
(one RM per agent).

Examples:
  - Built-in RM: `python frozen_lake_main.py`
  - Shared RM:   `python frozen_lake_main.py --rm-spec /tmp/rm_frozenlake.json`
  - Per-agent:   `python frozen_lake_main.py --rm-spec-a1 /tmp/a1.json --rm-spec-a2 /tmp/a2.json`
"""

from __future__ import annotations

import argparse
import copy

import numpy as np
import wandb

from multiagent_rlrm.environments.frozen_lake.action_encoder_frozen_lake import (
    ActionEncoderFrozenLake,
)
from multiagent_rlrm.environments.frozen_lake.detect_event import PositionEventDetector
from multiagent_rlrm.environments.frozen_lake.ma_frozen_lake import MultiAgentFrozenLake
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
from multiagent_rlrm.multi_agent.agent_rl import AgentRL
from multiagent_rlrm.multi_agent.reward_machine import RewardMachine
from multiagent_rlrm.multi_agent.wrappers.rm_environment_wrapper import (
    RMEnvironmentWrapper,
)
from multiagent_rlrm.render.heatmap import generate_heatmaps_for_agents
from multiagent_rlrm.render.render import EnvironmentRenderer
from multiagent_rlrm.utils.utils import parse_map_emoji

from multiagent_rlrm.environments.frozen_lake.config_frozen_lake import (
    config as frozenlake_config,
)

DEFAULT_NUM_EPISODES = 30_000
DEFAULT_RENDER_EVERY = 100
DEFAULT_SEED = 111
DEFAULT_MAP_NAME = "map1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-agent FrozenLake (RM) training")
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--map",
        type=str,
        default=DEFAULT_MAP_NAME,
        help="FrozenLake map name from config_frozen_lake.py (default: map1).",
    )
    train_group.add_argument(
        "--num-episodes",
        type=int,
        default=DEFAULT_NUM_EPISODES,
        help="Number of training episodes.",
    )
    train_group.add_argument(
        "--render-every",
        type=int,
        default=DEFAULT_RENDER_EVERY,
        help="Record a video every N episodes; set 0 to disable.",
    )
    train_group.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed used for env reset.",
    )

    rm_group = parser.add_argument_group("Reward Machine (optional)")
    rm_group.add_argument(
        "--rm-spec",
        dest="rm_spec",
        type=str,
        default=None,
        help="Path to an RM spec file (.json/.yaml). If set, overrides the built-in RM for all agents.",
    )
    rm_group.add_argument(
        "--rm-spec-a1",
        dest="rm_spec_a1",
        type=str,
        default=None,
        help="RM spec path for agent a1 (overrides --rm-spec).",
    )
    rm_group.add_argument(
        "--rm-spec-a2",
        dest="rm_spec_a2",
        type=str,
        default=None,
        help="RM spec path for agent a2 (overrides --rm-spec).",
    )
    rm_group.add_argument(
        "--complete-missing-transitions",
        action="store_true",
        help="Auto-complete missing RM transitions with self-loops and default reward (only with --rm-spec*).",
    )
    rm_group.add_argument(
        "--default-reward",
        type=float,
        default=0.0,
        help="Reward value for auto-completed transitions (only with --rm-spec*).",
    )

    return parser.parse_args()


def _build_frozenlake_event_mapping(goals):
    mapping = {}
    for label, pos in goals.items():
        mapping[f"at({label})"] = pos
        mapping[label] = pos
    return mapping


def _load_frozenlake_rm_from_spec(
    *,
    rm_spec_path: str,
    map_name: str,
    goals,
    agent_name: str,
    complete_missing_transitions: bool,
    default_reward: float,
) -> RewardMachine:
    from multiagent_rlrm.environments.frozen_lake.event_context import (
        build_frozenlake_context,
    )
    from multiagent_rlrm.rmgen.io import compile_reward_machine, load_rmspec
    from multiagent_rlrm.rmgen.normalize import enforce_env_id, normalize_rmspec_events
    from multiagent_rlrm.rmgen.summary import format_rmspec_summary
    from multiagent_rlrm.rmgen.validator import ValidationError

    try:
        spec = load_rmspec(rm_spec_path)
        spec = enforce_env_id(spec, "frozenlake", reason="--rm-spec is set")

        context = build_frozenlake_context(map_name)
        spec = normalize_rmspec_events(spec, context)

        event_mapping = _build_frozenlake_event_mapping(goals)
        event_detector_positions = set(goals.values())
        event_detector = PositionEventDetector(event_detector_positions)

        rm_compiled = compile_reward_machine(
            spec,
            event_detector=event_detector,
            event_mapping=event_mapping,
            complete_missing_transitions=complete_missing_transitions,
            default_reward=default_reward,
        )
        print(
            "\n"
            + format_rmspec_summary(
                spec,
                agent_names=[agent_name],
                source=str(rm_spec_path),
            )
            + "\n"
        )
        return rm_compiled
    except FileNotFoundError as exc:
        raise SystemExit(f"--rm-spec file not found: {exc}") from exc
    except ValidationError as exc:
        raise SystemExit(f"--rm-spec validation failed: {exc}") from exc
    except ValueError as exc:
        raise SystemExit(f"--rm-spec invalid: {exc}") from exc


def main() -> None:
    args = parse_args()
    render_every = None if args.render_every <= 0 else args.render_every

    # WandB is disabled; keep init for a consistent logging interface
    wandb.init(project="deep_FL", entity="alee8", mode="disabled")

    maps = frozenlake_config.get("maps", {})
    if args.map not in maps:
        raise SystemExit(f"Unknown map '{args.map}'. Available: {sorted(maps.keys())}")

    map_layout = maps[args.map]["layout"]

    # --- Environment and agents ----------------------------------------------- #
    holes, goals, dimensions = parse_map_emoji(map_layout)
    object_positions = {
        "holes": holes,
        "use_ice_background": True,
        "show_rm_panel": True,
    }

    env = MultiAgentFrozenLake(
        width=dimensions[0],
        height=dimensions[1],
        holes=holes,
    )
    env.frozen_lake_stochastic = False
    env.penalty_amount = 0
    env.delay_action = False

    a1 = AgentRL("a1", env)
    a2 = AgentRL("a2", env)

    a1.set_initial_position(5, 0)
    a2.set_initial_position(0, 0)

    for ag in (a1, a2):
        ag.add_state_encoder(StateEncoderFrozenLake(ag))
        ag.add_action_encoder(ActionEncoderFrozenLake(ag))

    event_detector = PositionEventDetector(set(goals.values()))

    rm_spec_a1 = args.rm_spec_a1 or args.rm_spec
    rm_spec_a2 = args.rm_spec_a2 or args.rm_spec
    if (rm_spec_a1 is None) != (rm_spec_a2 is None):
        raise SystemExit(
            "Provide either --rm-spec (shared) or both --rm-spec-a1 and --rm-spec-a2."
        )

    if rm_spec_a1 and rm_spec_a2:
        rm_q = _load_frozenlake_rm_from_spec(
            rm_spec_path=rm_spec_a1,
            map_name=args.map,
            goals=goals,
            agent_name="a1",
            complete_missing_transitions=args.complete_missing_transitions,
            default_reward=args.default_reward,
        )
        rm_qr = _load_frozenlake_rm_from_spec(
            rm_spec_path=rm_spec_a2,
            map_name=args.map,
            goals=goals,
            agent_name="a2",
            complete_missing_transitions=args.complete_missing_transitions,
            default_reward=args.default_reward,
        )
    else:
        # Built-in RM: reach A -> B -> C
        transitions = {
            ("state0", goals["A"]): ("state1", 10),
            ("state1", goals["B"]): ("state2", 15),
            ("state2", goals["C"]): ("state3", 20),
        }
        rm_q = RewardMachine(transitions, event_detector)
        rm_qr = RewardMachine(transitions, event_detector)

    a1.set_reward_machine(rm_q)
    a2.set_reward_machine(rm_qr)

    env.add_agent(a1)
    env.add_agent(a2)
    rm_env = RMEnvironmentWrapper(env, [a1, a2])

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

    rm_env.reset(args.seed)
    a1.get_learning_algorithm().learn_init()
    a2.get_learning_algorithm().learn_init()

    def record_greedy_episode(*, tag: str, seed: int) -> None:
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

    for episode in range(args.num_episodes):
        states, infos = rm_env.reset(args.seed)
        done = {a.name: False for a in rm_env.agents}
        rewards_agents = {a.name: 0 for a in rm_env.agents}
        record_episode = bool(render_every) and episode % render_every == 0

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

        if render_every and episode > 0 and episode % render_every == 0:
            print(f"Recording greedy episode at training episode {episode}...")
            record_greedy_episode(tag=f"greedy_ep_{episode}", seed=args.seed)

    save_q_tables(rm_env.agents)
    data = np.load("data/q_tables.npz")
    generate_heatmaps_for_agents(
        rm_env.agents, data, grid_dims=(env.grid_height, env.grid_width)
    )


if __name__ == "__main__":
    main()
