import argparse
import contextlib
import io
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from multiagent_rlrm.learning_algorithms.qlearning import QLearning
from multiagent_rlrm.learning_algorithms.qrmax_v2 import QRMax_v2
from multiagent_rlrm.learning_algorithms.rmax import RMax
from multiagent_rlrm.multi_agent.bucket_state_encoder import BucketStateEncoder
from multiagent_rlrm.multi_agent.reward_machine import RewardMachine


DEFAULT_EVENT_RECTS = {
    "A": (0.35, 0.45, 0.30, 0.70),
    "B": (0.55, 0.65, 0.30, 0.70),
    "C": (0.75, 0.85, 0.30, 0.70),
}

LAYOUT_OBSTACLES = {
    "open": (),
    "bottleneck": (
        (0.475, 0.525, 0.00, 0.40),
        (0.475, 0.525, 0.60, 1.00),
    ),
}


def _normalize_sequence(sequence: Union[Sequence[str], str]) -> Tuple[str, ...]:
    if isinstance(sequence, str):
        if "," in sequence:
            labels = tuple(label.strip().upper() for label in sequence.split(","))
        else:
            labels = tuple(sequence.upper())
    else:
        labels = tuple(str(label).upper() for label in sequence)
    if not labels:
        raise ValueError("event_sequence must contain at least one event label")
    unknown = [label for label in labels if label not in DEFAULT_EVENT_RECTS]
    if unknown:
        raise ValueError(f"Unsupported event labels: {unknown}")
    return labels


def _obstacles_for_layout(layout: str):
    if layout not in LAYOUT_OBSTACLES:
        raise ValueError(f"Unsupported layout {layout!r}")
    return LAYOUT_OBSTACLES[layout]


def _edges_with_boundaries(low, high, buckets, boundaries):
    edges = {float(low), float(high)}
    for idx in range(1, buckets):
        edges.add(round(float(low + (high - low) * idx / buckets), 6))
    for boundary in boundaries:
        if low < boundary < high:
            edges.add(round(float(boundary), 6))
    return tuple(sorted(edges))


def event_aligned_bucket_edges(event_detector, buckets_x, buckets_y):
    x_boundaries = []
    y_boundaries = []
    for rect in event_detector.event_rects.values():
        xmin, xmax, ymin, ymax = rect
        x_boundaries.extend([xmin, xmax])
        y_boundaries.extend([ymin, ymax])
    return (
        _edges_with_boundaries(0.0, 1.0, buckets_x, x_boundaries),
        _edges_with_boundaries(0.0, 1.0, buckets_y, y_boundaries),
    )


def transition_probed_bucket_edges(
    event_detector,
    obstacle_rects,
    event_sequence,
    buckets_x,
    buckets_y,
    *,
    step_size=0.025,
    probe_resolution=41,
):
    x_boundaries = []
    y_boundaries = []
    for rect in event_detector.event_rects.values():
        xmin, xmax, ymin, ymax = rect
        x_boundaries.extend([xmin, xmax])
        y_boundaries.extend([ymin, ymax])

    xs = np.linspace(0.0, 1.0 - 1e-6, probe_resolution)
    ys = np.linspace(0.0, 1.0 - 1e-6, probe_resolution)
    x_change_counts = Counter()
    y_change_counts = Counter()

    def blocked_at(x, y, action):
        dx, dy = ContinuousCorridorSequence.ACTIONS[action]
        probe_env = ContinuousCorridorSequence(
            seed=0,
            step_size=step_size,
            noise_std=0.0,
            event_detector=event_detector,
            event_sequence=event_sequence,
            obstacle_rects=obstacle_rects,
        )
        probe_env.x = float(x)
        probe_env.y = float(y)
        state, _, _, _, _ = probe_env.step(action)
        expected = step_size if dx or dy else 0.0
        movement = abs(state["x"] - x) + abs(state["y"] - y)
        return expected > 0 and movement < 0.5 * expected

    def refine_boundary(axis, fixed_value, lo, hi, action):
        lo_blocked = (
            blocked_at(lo, fixed_value, action)
            if axis == "x"
            else blocked_at(fixed_value, lo, action)
        )
        hi_blocked = (
            blocked_at(hi, fixed_value, action)
            if axis == "x"
            else blocked_at(fixed_value, hi, action)
        )
        if lo_blocked == hi_blocked:
            return float((lo + hi) / 2)
        left, right = float(lo), float(hi)
        left_blocked = lo_blocked
        for _ in range(16):
            mid = (left + right) / 2
            mid_blocked = (
                blocked_at(mid, fixed_value, action)
                if axis == "x"
                else blocked_at(fixed_value, mid, action)
            )
            if mid_blocked == left_blocked:
                left = mid
            else:
                right = mid
        return float((left + right) / 2)

    for action, (dx, dy) in ContinuousCorridorSequence.ACTIONS.items():
        blocked = np.zeros((probe_resolution, probe_resolution), dtype=bool)
        for ix, x in enumerate(xs):
            for iy, y in enumerate(ys):
                blocked[ix, iy] = blocked_at(x, y, action)

        for ix in range(probe_resolution - 1):
            changed = blocked[ix, :] != blocked[ix + 1, :]
            for iy in np.where(changed)[0]:
                boundary = round(
                    refine_boundary("x", ys[iy], xs[ix], xs[ix + 1], action),
                    3,
                )
                x_change_counts[boundary] += 1
        for iy in range(probe_resolution - 1):
            changed = blocked[:, iy] != blocked[:, iy + 1]
            for ix in np.where(changed)[0]:
                boundary = round(
                    refine_boundary("y", xs[ix], ys[iy], ys[iy + 1], action),
                    3,
                )
                y_change_counts[boundary] += 1

    for boundary, count in x_change_counts.items():
        if 0.05 < boundary < 0.95 and count >= 3:
            x_boundaries.append(boundary)
    for boundary, count in y_change_counts.items():
        if 0.05 < boundary < 0.95 and count >= 3:
            y_boundaries.append(boundary)

    return (
        _edges_with_boundaries(0.0, 1.0, buckets_x, x_boundaries),
        _edges_with_boundaries(0.0, 1.0, buckets_y, y_boundaries),
    )


class CorridorEventDetector:
    """Detect RM events in rectangular regions of a 2D continuous corridor."""

    def __init__(
        self,
        a_rect: Tuple[float, float, float, float] = (0.35, 0.45, 0.30, 0.70),
        b_rect: Tuple[float, float, float, float] = (0.55, 0.65, 0.30, 0.70),
        c_rect: Tuple[float, float, float, float] = (0.75, 0.85, 0.30, 0.70),
        *,
        event_labels: Sequence[str] = ("A", "B"),
        event_rects: Optional[Mapping[str, Tuple[float, float, float, float]]] = None,
    ):
        labels = _normalize_sequence(event_labels)
        if event_rects is None:
            rects = {
                "A": a_rect,
                "B": b_rect,
                "C": c_rect,
            }
        else:
            rects = dict(event_rects)
        self.event_rects = {label: rects[label] for label in labels}
        self.event_labels = tuple(self.event_rects)

    @staticmethod
    def _contains(rect, x, y):
        xmin, xmax, ymin, ymax = rect
        return xmin <= x <= xmax and ymin <= y <= ymax

    def detect_event(self, state):
        x = state["x"]
        y = state["y"]
        for label, rect in self.event_rects.items():
            if self._contains(rect, x, y):
                return label
        return None


class ContinuousCorridorAgent:
    """Minimal agent API needed by BucketStateEncoder."""

    def __init__(self, event_detector, event_sequence: Sequence[str] = ("A", "B")):
        labels = _normalize_sequence(event_sequence)
        transitions = {}
        for q_idx, label in enumerate(labels):
            current_q = _rm_state_name(q_idx)
            next_q = _rm_state_name(q_idx + 1)
            transitions[(current_q, None)] = (current_q, 0)
            transitions[(current_q, label)] = (
                next_q,
                1 if q_idx == len(labels) - 1 else 0,
            )
        final_q = _rm_state_name(len(labels))
        transitions[(final_q, None)] = (final_q, 0)
        self._reward_machine = RewardMachine(transitions, event_detector)

    def get_reward_machine(self):
        return self._reward_machine


class ContinuousCorridorSequence:
    """A 2D continuous-state NMRDP with sequential RM events."""

    ACTIONS = {
        0: (-1.0, 0.0),
        1: (1.0, 0.0),
        2: (0.0, -1.0),
        3: (0.0, 1.0),
    }

    def __init__(
        self,
        *,
        seed: int,
        event_detector: Optional[CorridorEventDetector] = None,
        step_size: float = 0.025,
        noise_std: float = 0.003,
        horizon: int = 80,
        reset_x_range: Tuple[float, float] = (0.45, 0.55),
        reset_y_range: Tuple[float, float] = (0.42, 0.58),
        event_sequence: Sequence[str] = ("A", "B"),
        obstacle_rects: Sequence[Tuple[float, float, float, float]] = (),
    ):
        self.rng = np.random.default_rng(seed)
        self.event_sequence = _normalize_sequence(event_sequence)
        self.event_detector = event_detector or CorridorEventDetector(
            event_labels=self.event_sequence
        )
        self.step_size = step_size
        self.noise_std = noise_std
        self.horizon = horizon
        self.reset_x_range = reset_x_range
        self.reset_y_range = reset_y_range
        self.obstacle_rects = tuple(obstacle_rects)
        self.x = 0.5
        self.y = 0.5
        self.q_idx = 0
        self.steps = 0

    def reset(self):
        for _ in range(1000):
            x = float(self.rng.uniform(*self.reset_x_range))
            y = float(self.rng.uniform(*self.reset_y_range))
            if not self._is_blocked(x, y):
                self.x = x
                self.y = y
                break
        else:
            raise ValueError("reset range only samples obstacle cells")
        self.q_idx = 0
        self.steps = 0
        return {"x": self.x, "y": self.y}, self.q_idx

    def _is_blocked(self, x, y):
        return any(
            CorridorEventDetector._contains(rect, x, y)
            for rect in self.obstacle_rects
        )

    def transition_from(self, q_idx, event):
        if q_idx >= len(self.event_sequence):
            return q_idx, 0.0, True
        if event == self.event_sequence[q_idx]:
            next_q_idx = q_idx + 1
            solved = next_q_idx == len(self.event_sequence)
            return next_q_idx, 1.0 if solved else 0.0, solved
        return q_idx, 0.0, False

    def step(self, action):
        dx, dy = self.ACTIONS[action]
        noise_x = (
            float(self.rng.normal(0.0, self.noise_std))
            if self.noise_std
            else 0.0
        )
        noise_y = (
            float(self.rng.normal(0.0, self.noise_std))
            if self.noise_std
            else 0.0
        )
        next_x = min(0.999999, max(0.0, self.x + dx * self.step_size + noise_x))
        next_y = min(0.999999, max(0.0, self.y + dy * self.step_size + noise_y))
        if not self._is_blocked(next_x, next_y):
            self.x = next_x
            self.y = next_y
        event = self.event_detector.detect_event({"x": self.x, "y": self.y})
        self.q_idx, reward, solved = self.transition_from(self.q_idx, event)
        self.steps += 1
        truncated = self.steps >= self.horizon
        return {"x": self.x, "y": self.y}, self.q_idx, reward, solved or truncated, solved


@dataclass(frozen=True)
class BucketQRMaxTrialResult:
    seed: int
    algorithm: str
    include_event_label: bool
    success_rate: float
    average_length: Optional[float]
    first_perfect_eval_episode: Optional[int]
    training_successes: int
    event_counts: Dict[str, int]


def make_encoder(
    *,
    include_event_label: bool,
    buckets_x: int = 10,
    buckets_y: int = 4,
    event_detector: Optional[CorridorEventDetector] = None,
    event_sequence: Sequence[str] = ("A", "B"),
    bucket_mode: str = "uniform",
    obstacle_rects: Sequence[Tuple[float, float, float, float]] = (),
):
    labels = _normalize_sequence(event_sequence)
    detector = event_detector or CorridorEventDetector(event_labels=labels)
    if bucket_mode == "uniform":
        bucket_edges = None
    elif bucket_mode == "event-aligned":
        bucket_edges = event_aligned_bucket_edges(detector, buckets_x, buckets_y)
    elif bucket_mode == "transition-probed":
        bucket_edges = transition_probed_bucket_edges(
            detector,
            obstacle_rects,
            labels,
            buckets_x,
            buckets_y,
        )
    else:
        raise ValueError(f"Unsupported bucket_mode {bucket_mode!r}")
    return BucketStateEncoder(
        ContinuousCorridorAgent(detector, event_sequence=labels),
        features=["x", "y"],
        lows=[0.0, 0.0],
        highs=[1.0, 1.0],
        buckets=[buckets_x, buckets_y],
        bucket_edges=bucket_edges,
        include_event_label=include_event_label,
        event_detector=detector,
        event_labels=labels,
    )


def _rm_state_name(q_idx):
    return f"q{q_idx}"


def _evaluate_policy(algo, encoder, env, episodes):
    successes = 0
    lengths = []
    for _ in range(episodes):
        state, q_idx = env.reset()
        for step in range(env.horizon):
            encoded_state, info = encoder.encode(state, _rm_state_name(q_idx))
            action = algo.choose_action(encoded_state, best=True, info=info)
            state, q_idx, _, stopped, solved = env.step(action)
            if stopped:
                if solved:
                    successes += 1
                    lengths.append(step + 1)
                break
    return successes / episodes, statistics.mean(lengths) if lengths else None


def _make_algorithm(algorithm, encoder, threshold, seed):
    state_space_size = encoder.total_state_space_size
    q_space_size = encoder.agent.get_reward_machine().numbers_state()
    if algorithm == "QRMAX":
        with contextlib.redirect_stdout(io.StringIO()):
            algo = QRMax_v2(
                state_space_size=state_space_size,
                action_space_size=4,
                q_space_size=q_space_size,
                gamma=0.95,
                max_reward=1.0,
                nsamplesTE=threshold,
                nsamplesRE=threshold,
                nsamplesTQ=1,
                nsamplesRQ=1,
                VI_delta=1e-3,
                VI_delta_rel=True,
                seed=seed,
            )
        algo.max_num_value_iter = 600
        algo.learn_init()
        return algo
    if algorithm in {"QL", "QRM"}:
        return QLearning(
            state_space_size=state_space_size,
            action_space_size=4,
            gamma=0.95,
            action_selection="greedy",
            learning_rate=0.1,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.995,
            qtable_init=0,
            use_qrm=algorithm == "QRM",
            seed=seed,
        )
    if algorithm == "RMAX":
        with contextlib.redirect_stdout(io.StringIO()):
            algo = RMax(
                state_space_size=state_space_size,
                action_space_size=4,
                gamma=0.95,
                max_reward=1.0,
                s_a_threshold=threshold,
                VI_delta=1e-3,
                VI_delta_rel=True,
                seed=seed,
            )
        algo.max_num_value_iter = 600
        return algo
    raise ValueError(f"Unsupported algorithm {algorithm!r}")


def _qrm_experiences(encoder, env, state, next_state, action, event):
    experiences = []
    rm = encoder.agent.get_reward_machine()
    for q_idx in range(rm.numbers_state()):
        next_q_idx, reward, solved = env.transition_from(q_idx, event)
        encoded_state, _ = encoder.encode(state, _rm_state_name(q_idx))
        encoded_next_state, _ = encoder.encode(next_state, _rm_state_name(next_q_idx))
        experiences.append(
            (
                encoded_state,
                action,
                reward,
                encoded_next_state,
                solved,
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
    buckets_x: int = 10,
    buckets_y: int = 4,
    threshold: int = 8,
    train_episodes: int = 300,
    eval_episodes: int = 100,
    eval_interval: int = 25,
    step_size: float = 0.025,
    noise_std: float = 0.003,
    horizon: int = 80,
    reset_x_low: float = 0.45,
    reset_x_high: float = 0.55,
    reset_y_low: float = 0.42,
    reset_y_high: float = 0.58,
    event_sequence: Sequence[str] = ("A", "B"),
    layout: str = "open",
    bucket_mode: str = "uniform",
) -> BucketQRMaxTrialResult:
    labels = _normalize_sequence(event_sequence)
    obstacle_rects = _obstacles_for_layout(layout)
    event_detector = CorridorEventDetector(event_labels=labels)
    encoder = make_encoder(
        include_event_label=include_event_label,
        buckets_x=buckets_x,
        buckets_y=buckets_y,
        event_detector=event_detector,
        event_sequence=labels,
        bucket_mode=bucket_mode,
        obstacle_rects=obstacle_rects,
    )
    env = ContinuousCorridorSequence(
        seed=seed,
        event_detector=event_detector,
        step_size=step_size,
        noise_std=noise_std,
        horizon=horizon,
        reset_x_range=(reset_x_low, reset_x_high),
        reset_y_range=(reset_y_low, reset_y_high),
        event_sequence=labels,
        obstacle_rects=obstacle_rects,
    )
    algo = _make_algorithm(algorithm, encoder, threshold, seed)

    event_counts = Counter({label: 0 for label in labels})
    training_successes = 0
    first_perfect_eval_episode = None
    for episode in range(train_episodes):
        state, q_idx = env.reset()
        algo.learn_init_episode()
        for _ in range(horizon):
            encoded_state, info = encoder.encode(state, _rm_state_name(q_idx))
            action = algo.choose_action(encoded_state, best=False, info=info)
            next_state, next_q_idx, reward, stopped, solved = env.step(action)
            event = event_detector.detect_event(next_state)
            if event is not None:
                event_counts[event] += 1
            if solved:
                training_successes += 1
            encoded_next_state, next_info = encoder.encode(
                next_state, _rm_state_name(next_q_idx)
            )
            update_info = {
                "prev_s": info["s"],
                "prev_q": q_idx,
                "s": next_info["s"],
                "q": next_q_idx,
                "Renv": 0.0,
                "RQ": reward,
                "qrm_experience": _qrm_experiences(
                    encoder, env, state, next_state, action, event
                ),
            }
            with contextlib.redirect_stdout(io.StringIO()):
                algo.update(
                    encoded_state,
                    encoded_next_state,
                    action,
                    reward,
                    solved,
                    info=update_info,
                )
            state, q_idx = next_state, next_q_idx
            if stopped:
                break
        algo.learn_done_episode()

        if first_perfect_eval_episode is None and (episode + 1) % eval_interval == 0:
            eval_env = ContinuousCorridorSequence(
                seed=seed + 100000 + episode,
                event_detector=event_detector,
                step_size=step_size,
                noise_std=noise_std,
                horizon=horizon,
                reset_x_range=(reset_x_low, reset_x_high),
                reset_y_range=(reset_y_low, reset_y_high),
                event_sequence=labels,
                obstacle_rects=obstacle_rects,
            )
            success_rate, _ = _evaluate_policy(algo, encoder, eval_env, 20)
            if success_rate == 1.0:
                first_perfect_eval_episode = episode + 1

    eval_env = ContinuousCorridorSequence(
        seed=seed + 200000,
        event_detector=event_detector,
        step_size=step_size,
        noise_std=noise_std,
        horizon=horizon,
        reset_x_range=(reset_x_low, reset_x_high),
        reset_y_range=(reset_y_low, reset_y_high),
        event_sequence=labels,
        obstacle_rects=obstacle_rects,
    )
    success_rate, average_length = _evaluate_policy(
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
        result.average_length
        for result in results
        if result.average_length is not None
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
        description="Run the continuous-corridor Bucket QR-MAX experiment."
    )
    parser.add_argument(
        "--algorithm", choices=["QRMAX", "QL", "QRM", "RMAX"], default="QRMAX"
    )
    parser.add_argument("--seeds", default="1700:1705")
    parser.add_argument("--sequence", choices=["AB", "ABC"], default="AB")
    parser.add_argument("--layout", choices=sorted(LAYOUT_OBSTACLES), default="open")
    parser.add_argument(
        "--bucket-mode",
        choices=[
            "uniform",
            "event-aligned",
            "transition-probed",
        ],
        default="uniform",
    )
    parser.add_argument("--buckets-x", type=int, default=10)
    parser.add_argument("--buckets-y", type=int, default=4)
    parser.add_argument("--threshold", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=0.003)
    parser.add_argument("--reset-x-low", type=float, default=0.45)
    parser.add_argument("--reset-x-high", type=float, default=0.55)
    parser.add_argument("--reset-y-low", type=float, default=0.42)
    parser.add_argument("--reset-y-high", type=float, default=0.58)
    parser.add_argument("--horizon", type=int, default=80)
    parser.add_argument("--train-episodes", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument(
        "--event-aware",
        action="store_true",
        help="Append the detected RM event label to the bucket state.",
    )
    args = parser.parse_args(argv)

    summary = run_bucket_qrmax_sweep(
        seeds=_parse_seed_range(args.seeds),
        algorithm=args.algorithm,
        include_event_label=args.event_aware,
        buckets_x=args.buckets_x,
        buckets_y=args.buckets_y,
        threshold=args.threshold,
        noise_std=args.noise_std,
        event_sequence=args.sequence,
        layout=args.layout,
        bucket_mode=args.bucket_mode,
        reset_x_low=args.reset_x_low,
        reset_x_high=args.reset_x_high,
        reset_y_low=args.reset_y_low,
        reset_y_high=args.reset_y_high,
        horizon=args.horizon,
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
    )
    avg_len = summary["average_length_mean"]
    avg_len_text = "NA" if avg_len is None else f"{avg_len:.3f}"
    print(
        "algorithm,event_aware,sequence,layout,bucket_mode,buckets_x,buckets_y,threshold,"
        "noise,seeds,success_mean,success_min,success_max,avg_len,"
        "training_successes_mean,event_count_means"
    )
    print(
        f"{summary['algorithm']},{summary['include_event_label']},"
        f"{args.sequence},{args.layout},{args.bucket_mode},"
        f"{args.buckets_x},{args.buckets_y},{args.threshold},{args.noise_std},"
        f"{summary['num_seeds']},{summary['success_mean']:.3f},"
        f"{summary['success_min']:.3f},{summary['success_max']:.3f},"
        f"{avg_len_text},{summary['training_successes_mean']:.3f},"
        f"{summary['event_count_means']}"
    )


if __name__ == "__main__":
    main()
