import argparse
import contextlib
import io
import json
import statistics
from collections import Counter
from dataclasses import dataclass
from math import floor
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from multiagent_rlrm.environments.frozen_lake.config_frozen_lake import (
    config as frozenlake_config,
)
from multiagent_rlrm.learning_algorithms.qlearning import QLearning
from multiagent_rlrm.learning_algorithms.qrmax_v2 import QRMax_v2
from multiagent_rlrm.learning_algorithms.rmax import RMax
from multiagent_rlrm.multi_agent.bucket_state_encoder import BucketStateEncoder
from multiagent_rlrm.multi_agent.reward_machine import RewardMachine
from multiagent_rlrm.utils.utils import parse_map_emoji


ACTION_DELTAS = {
    0: (0.0, -1.0),
    1: (0.0, 1.0),
    2: (-1.0, 0.0),
    3: (1.0, 0.0),
}
SEQUENCES = {
    "AB": ("A", "B"),
    "ABC": ("A", "B", "C"),
}


def _normalize_sequence(sequence: str | Sequence[str]) -> Tuple[str, ...]:
    if isinstance(sequence, str):
        if sequence in SEQUENCES:
            return SEQUENCES[sequence]
        return tuple(sequence)
    return tuple(sequence)


def _cell_from_state(state, width, height):
    x = min(width - 1, max(0, int(floor(state["x"]))))
    y = min(height - 1, max(0, int(floor(state["y"]))))
    return x, y


class FrozenLakeEventDetector:
    """Detect RM events and terminal holes from continuous coordinates."""

    def __init__(self, *, goals, holes, width, height):
        self.goals_by_cell = {position: label for label, position in goals.items()}
        self.holes = set(holes)
        self.width = width
        self.height = height

    def detect_event(self, state):
        cell = _cell_from_state(state, self.width, self.height)
        if cell in self.holes:
            return "hole"
        return self.goals_by_cell.get(cell)


class ContinuousFrozenLakeAgent:
    """Minimal agent API needed by the bucket encoders."""

    def __init__(self, event_detector, event_sequence):
        labels = _normalize_sequence(event_sequence)
        transitions = {}
        for idx, label in enumerate(labels):
            reward = 1.0 if idx == len(labels) - 1 else 0.0
            transitions[(f"q{idx}", label)] = (f"q{idx + 1}", reward)
        transitions[(f"q{len(labels)}", None)] = (f"q{len(labels)}", 0.0)
        self._reward_machine = RewardMachine(transitions, event_detector)

    def get_reward_machine(self):
        return self._reward_machine


class ContinuousFrozenLakeSequence:
    """Continuous-coordinate Frozen Lake with sequential RM events."""

    def __init__(
        self,
        *,
        seed: int,
        goals,
        holes,
        width: int,
        height: int,
        event_detector: FrozenLakeEventDetector,
        event_sequence: Sequence[str] = ("A", "B", "C"),
        start_cell: Tuple[int, int] = (5, 0),
        step_size: float = 0.65,
        noise_std: float = 0.05,
        slip_probability: float = 0.10,
        horizon: int = 160,
    ):
        self.rng = np.random.default_rng(seed)
        self.goals = goals
        self.holes = set(holes)
        self.width = width
        self.height = height
        self.event_detector = event_detector
        self.event_sequence = _normalize_sequence(event_sequence)
        self.start_cell = start_cell
        self.step_size = step_size
        self.noise_std = noise_std
        self.slip_probability = slip_probability
        self.horizon = horizon
        self.x = 0.0
        self.y = 0.0
        self.q_idx = 0
        self.steps = 0

    def reset(self):
        self.x = float(self.start_cell[0] + 0.5 + self.rng.uniform(-0.10, 0.10))
        self.y = float(self.start_cell[1] + 0.5 + self.rng.uniform(-0.10, 0.10))
        self.q_idx = 0
        self.steps = 0
        return self.state(), self.q_idx

    def state(self):
        return {"x": self.x, "y": self.y}

    def current_cell(self):
        return _cell_from_state(self.state(), self.width, self.height)

    def transition_from(self, q_idx, event):
        final_q = len(self.event_sequence)
        if q_idx >= final_q:
            return final_q, 0.0, True
        expected_event = self.event_sequence[q_idx]
        if event == expected_event:
            next_q = q_idx + 1
            solved = next_q == final_q
            return next_q, 1.0 if solved else 0.0, solved
        return q_idx, 0.0, False

    def step(self, action):
        if self.slip_probability and self.rng.uniform() < self.slip_probability:
            action = int(self.rng.integers(0, len(ACTION_DELTAS)))
        dx, dy = ACTION_DELTAS[action]
        noise_x = float(self.rng.normal(0.0, self.noise_std)) if self.noise_std else 0.0
        noise_y = float(self.rng.normal(0.0, self.noise_std)) if self.noise_std else 0.0
        self.x = min(
            self.width - 1e-6,
            max(0.0, self.x + dx * self.step_size + noise_x),
        )
        self.y = min(
            self.height - 1e-6,
            max(0.0, self.y + dy * self.step_size + noise_y),
        )
        state = self.state()
        event = self.event_detector.detect_event(state)
        failed = event == "hole"
        self.q_idx, reward, solved = self.transition_from(self.q_idx, event)
        self.steps += 1
        truncated = self.steps >= self.horizon
        stopped = solved or failed or truncated
        return state, self.q_idx, reward, stopped, solved, failed, event


@dataclass(frozen=True)
class BucketQRMaxTrialResult:
    seed: int
    algorithm: str
    include_event_label: bool
    success_rate: float
    average_length: Optional[float]
    first_perfect_eval_episode: Optional[int]
    training_successes: int
    training_failures: int
    event_counts: Dict[str, int]


def _map_data(map_name):
    maps = frozenlake_config.get("maps", {})
    if map_name not in maps:
        raise ValueError(f"Unknown FrozenLake map {map_name!r}")
    holes, goals, dimensions = parse_map_emoji(maps[map_name]["layout"])
    return holes, goals, dimensions


def make_encoder(
    *,
    map_name: str = "map1",
    include_event_label: bool,
    event_sequence: Sequence[str] = ("A", "B", "C"),
    bucket_mode: str = "uniform",
    buckets_x: int = 10,
    buckets_y: int = 10,
    event_detector: Optional[FrozenLakeEventDetector] = None,
):
    holes, goals, dimensions = _map_data(map_name)
    width, height = dimensions
    labels = _normalize_sequence(event_sequence)
    detector = event_detector or FrozenLakeEventDetector(
        goals=goals,
        holes=holes,
        width=width,
        height=height,
    )
    event_labels = sorted(set(goals.keys()) | set(labels) | {"hole"})
    agent = ContinuousFrozenLakeAgent(detector, labels)
    if bucket_mode == "uniform":
        return BucketStateEncoder(
            agent,
            features=["x", "y"],
            lows=[0.0, 0.0],
            highs=[float(width), float(height)],
            buckets=[buckets_x, buckets_y],
            include_event_label=include_event_label,
            event_detector=detector,
            event_labels=event_labels,
        )
    raise ValueError(f"Unsupported bucket_mode {bucket_mode!r}")


def _rm_state_name(q_idx):
    return f"q{q_idx}"


def _make_env(
    *,
    seed,
    map_name,
    event_detector,
    event_sequence,
    start_cell,
    step_size,
    noise_std,
    slip_probability,
    horizon,
):
    holes, goals, dimensions = _map_data(map_name)
    return ContinuousFrozenLakeSequence(
        seed=seed,
        goals=goals,
        holes=holes,
        width=dimensions[0],
        height=dimensions[1],
        event_detector=event_detector,
        event_sequence=event_sequence,
        start_cell=start_cell,
        step_size=step_size,
        noise_std=noise_std,
        slip_probability=slip_probability,
        horizon=horizon,
    )


def _evaluate_policy(algo, encoder, env, episodes):
    successes = 0
    lengths = []
    failures = 0
    for _ in range(episodes):
        state, q_idx = env.reset()
        for step in range(env.horizon):
            encoded_state, info = encoder.encode(state, _rm_state_name(q_idx))
            action = algo.choose_action(encoded_state, best=True, info=info)
            state, q_idx, _, stopped, solved, failed, _ = env.step(action)
            if stopped:
                if solved:
                    successes += 1
                    lengths.append(step + 1)
                if failed:
                    failures += 1
                break
    return (
        successes / episodes,
        statistics.mean(lengths) if lengths else None,
        failures / episodes,
    )


def _make_algorithm(algorithm, encoder, threshold, seed):
    state_space_size = encoder.total_state_space_size
    q_space_size = encoder.agent.get_reward_machine().numbers_state()
    if algorithm == "QRMAX":
        with contextlib.redirect_stdout(io.StringIO()):
            algo = QRMax_v2(
                state_space_size=state_space_size,
                action_space_size=4,
                q_space_size=q_space_size,
                gamma=0.97,
                max_reward=1.0,
                nsamplesTE=threshold,
                nsamplesRE=threshold,
                nsamplesTQ=1,
                nsamplesRQ=1,
                VI_delta=1e-3,
                VI_delta_rel=True,
                seed=seed,
            )
        algo.max_num_value_iter = 500
        algo.learn_init()
        return algo
    if algorithm in {"QL", "QRM"}:
        return QLearning(
            state_space_size=state_space_size,
            action_space_size=4,
            gamma=0.97,
            action_selection="greedy",
            learning_rate=0.15,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.997,
            qtable_init=0,
            use_qrm=algorithm == "QRM",
            seed=seed,
        )
    if algorithm == "RMAX":
        with contextlib.redirect_stdout(io.StringIO()):
            algo = RMax(
                state_space_size=state_space_size,
                action_space_size=4,
                gamma=0.97,
                max_reward=1.0,
                s_a_threshold=threshold,
                VI_delta=1e-3,
                VI_delta_rel=True,
                seed=seed,
            )
        algo.max_num_value_iter = 500
        return algo
    raise ValueError(f"Unsupported algorithm {algorithm!r}")


def _qrm_experiences(encoder, env, state, next_state, action, event, failed):
    experiences = []
    rm = encoder.agent.get_reward_machine()
    for q_idx in range(rm.numbers_state()):
        next_q_idx, reward, solved = env.transition_from(q_idx, event)
        encoded_state, _ = encoder.encode(state, _rm_state_name(q_idx))
        encoded_next_state, _ = encoder.encode(next_state, _rm_state_name(next_q_idx))
        terminated = solved or failed
        experiences.append(
            (
                encoded_state,
                action,
                reward,
                encoded_next_state,
                terminated,
                None,
                q_idx,
                None,
                next_q_idx,
                reward,
            )
        )
    return experiences


def run_bucket_qrmax_trial(
    *,
    seed: int,
    include_event_label: bool,
    algorithm: str = "QRMAX",
    map_name: str = "map1",
    event_sequence: Sequence[str] = ("A", "B", "C"),
    bucket_mode: str = "uniform",
    buckets_x: int = 10,
    buckets_y: int = 10,
    threshold: int = 8,
    train_episodes: int = 600,
    eval_episodes: int = 100,
    eval_interval: int = 50,
    step_size: float = 0.65,
    noise_std: float = 0.05,
    slip_probability: float = 0.10,
    horizon: int = 160,
    start_x: int = 5,
    start_y: int = 0,
) -> BucketQRMaxTrialResult:
    holes, goals, dimensions = _map_data(map_name)
    labels = _normalize_sequence(event_sequence)
    event_detector = FrozenLakeEventDetector(
        goals=goals,
        holes=holes,
        width=dimensions[0],
        height=dimensions[1],
    )
    encoder = make_encoder(
        map_name=map_name,
        include_event_label=include_event_label,
        event_sequence=labels,
        bucket_mode=bucket_mode,
        buckets_x=buckets_x,
        buckets_y=buckets_y,
        event_detector=event_detector,
    )
    env_kwargs = {
        "map_name": map_name,
        "event_detector": event_detector,
        "event_sequence": labels,
        "start_cell": (start_x, start_y),
        "step_size": step_size,
        "noise_std": noise_std,
        "slip_probability": slip_probability,
        "horizon": horizon,
    }
    env = _make_env(seed=seed, **env_kwargs)
    algo = _make_algorithm(algorithm, encoder, threshold, seed)

    event_counts = Counter({label: 0 for label in [*labels, "hole"]})
    training_successes = 0
    training_failures = 0
    first_perfect_eval_episode = None
    for episode in range(train_episodes):
        state, q_idx = env.reset()
        algo.learn_init_episode()
        for _ in range(horizon):
            encoded_state, info = encoder.encode(state, _rm_state_name(q_idx))
            action = algo.choose_action(encoded_state, best=False, info=info)
            next_state, next_q_idx, reward, stopped, solved, failed, event = env.step(
                action
            )
            if event is not None:
                event_counts[event] += 1
            if solved:
                training_successes += 1
            if failed:
                training_failures += 1
            encoded_next_state, next_info = encoder.encode(
                next_state, _rm_state_name(next_q_idx)
            )
            terminated = stopped
            update_info = {
                "prev_s": info["s"],
                "prev_q": q_idx,
                "s": next_info["s"],
                "q": next_q_idx,
                "Renv": 0.0,
                "RQ": reward,
                "qrm_experience": _qrm_experiences(
                    encoder, env, state, next_state, action, event, failed
                ),
            }
            with contextlib.redirect_stdout(io.StringIO()):
                algo.update(
                    encoded_state,
                    encoded_next_state,
                    action,
                    reward,
                    terminated,
                    info=update_info,
                )
            state, q_idx = next_state, next_q_idx
            if stopped:
                break
        algo.learn_done_episode()

        if first_perfect_eval_episode is None and (episode + 1) % eval_interval == 0:
            eval_env = _make_env(seed=seed + 100000 + episode, **env_kwargs)
            success_rate, _, _ = _evaluate_policy(algo, encoder, eval_env, 20)
            if success_rate == 1.0:
                first_perfect_eval_episode = episode + 1

    eval_env = _make_env(seed=seed + 200000, **env_kwargs)
    success_rate, average_length, _ = _evaluate_policy(
        algo, encoder, eval_env, eval_episodes
    )
    return BucketQRMaxTrialResult(
        seed=seed,
        algorithm=algorithm,
        include_event_label=include_event_label,
        success_rate=success_rate,
        average_length=average_length,
        first_perfect_eval_episode=first_perfect_eval_episode,
        training_successes=training_successes,
        training_failures=training_failures,
        event_counts=dict(event_counts),
    )


def run_bucket_qrmax_sweep(
    *,
    seeds: Iterable[int],
    include_event_label: bool,
    algorithm: str = "QRMAX",
    **trial_kwargs,
) -> Dict[str, object]:
    results = [
        run_bucket_qrmax_trial(
            seed=seed,
            include_event_label=include_event_label,
            algorithm=algorithm,
            **trial_kwargs,
        )
        for seed in seeds
    ]
    success_rates = [result.success_rate for result in results]
    lengths = [
        result.average_length for result in results if result.average_length is not None
    ]
    first_perfect = [
        result.first_perfect_eval_episode
        for result in results
        if result.first_perfect_eval_episode is not None
    ]
    event_labels = sorted(
        {label for result in results for label in result.event_counts}
    )
    return {
        "include_event_label": include_event_label,
        "algorithm": algorithm,
        "num_seeds": len(results),
        "success_mean": statistics.mean(success_rates),
        "success_min": min(success_rates),
        "success_max": max(success_rates),
        "average_length_mean": statistics.mean(lengths) if lengths else None,
        "first_perfect_eval_episode_mean": (
            statistics.mean(first_perfect) if first_perfect else None
        ),
        "training_successes_mean": statistics.mean(
            result.training_successes for result in results
        ),
        "training_failures_mean": statistics.mean(
            result.training_failures for result in results
        ),
        "event_count_means": {
            label: statistics.mean(
                result.event_counts.get(label, 0) for result in results
            )
            for label in event_labels
        },
        "results": results,
    }


def _parse_seed_range(value):
    if ":" in value:
        start, stop = value.split(":", 1)
        return range(int(start), int(stop))
    return [int(part) for part in value.split(",") if part]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the continuous-FrozenLake Bucket QR-MAX experiment."
    )
    parser.add_argument(
        "--algorithm", choices=["QRMAX", "QL", "QRM", "RMAX"], default="QRMAX"
    )
    parser.add_argument("--map", default="map1")
    parser.add_argument("--seeds", default="2000:2005")
    parser.add_argument("--sequence", choices=sorted(SEQUENCES), default="ABC")
    parser.add_argument("--bucket-mode", choices=["uniform"], default="uniform")
    parser.add_argument("--buckets-x", type=int, default=10)
    parser.add_argument("--buckets-y", type=int, default=10)
    parser.add_argument("--threshold", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--slip-probability", type=float, default=0.10)
    parser.add_argument("--step-size", type=float, default=0.65)
    parser.add_argument("--horizon", type=int, default=160)
    parser.add_argument("--train-episodes", type=int, default=600)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--start-x", type=int, default=5)
    parser.add_argument("--start-y", type=int, default=0)
    parser.add_argument(
        "--event-aware",
        action="store_true",
        help="Append detected goal/hole labels to the abstract state.",
    )
    args = parser.parse_args(argv)

    summary = run_bucket_qrmax_sweep(
        seeds=_parse_seed_range(args.seeds),
        algorithm=args.algorithm,
        include_event_label=args.event_aware,
        map_name=args.map,
        event_sequence=args.sequence,
        bucket_mode=args.bucket_mode,
        buckets_x=args.buckets_x,
        buckets_y=args.buckets_y,
        threshold=args.threshold,
        noise_std=args.noise_std,
        slip_probability=args.slip_probability,
        step_size=args.step_size,
        horizon=args.horizon,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        start_x=args.start_x,
        start_y=args.start_y,
    )
    avg_len = summary["average_length_mean"]
    avg_len_text = "NA" if avg_len is None else f"{avg_len:.3f}"
    print(
        "algorithm,event_aware,map,sequence,bucket_mode,buckets_x,buckets_y,"
        "threshold,noise,slip,seeds,success_mean,success_min,success_max,avg_len,"
        "first_perfect_eval_episode_mean,training_successes_mean,"
        "training_failures_mean,event_count_means"
    )
    print(
        f"{summary['algorithm']},{summary['include_event_label']},"
        f"{args.map},{args.sequence},{args.bucket_mode},"
        f"{args.buckets_x},{args.buckets_y},{args.threshold},"
        f"{args.noise_std},{args.slip_probability},"
        f"{summary['num_seeds']},{summary['success_mean']:.3f},"
        f"{summary['success_min']:.3f},{summary['success_max']:.3f},"
        f"{avg_len_text},{summary['first_perfect_eval_episode_mean']},"
        f"{summary['training_successes_mean']:.3f},"
        f"{summary['training_failures_mean']:.3f},"
        f"{json.dumps(summary['event_count_means'], sort_keys=True)}"
    )


if __name__ == "__main__":
    main()
