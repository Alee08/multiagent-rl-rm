import argparse
import contextlib
import io
import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from multiagent_rlrm.learning_algorithms.qlearning import QLearning
from multiagent_rlrm.learning_algorithms.qrmax_v2 import QRMax_v2
from multiagent_rlrm.learning_algorithms.rmax import RMax
from multiagent_rlrm.multi_agent.bucket_state_encoder import BucketStateEncoder
from multiagent_rlrm.multi_agent.reward_machine import RewardMachine


class CorridorEventDetector:
    """Detect A/B events in rectangular regions of a 2D continuous corridor."""

    def __init__(
        self,
        a_rect: Tuple[float, float, float, float] = (0.35, 0.45, 0.30, 0.70),
        b_rect: Tuple[float, float, float, float] = (0.55, 0.65, 0.30, 0.70),
    ):
        self.a_rect = a_rect
        self.b_rect = b_rect

    @staticmethod
    def _contains(rect, x, y):
        xmin, xmax, ymin, ymax = rect
        return xmin <= x <= xmax and ymin <= y <= ymax

    def detect_event(self, state):
        x = state["x"]
        y = state["y"]
        if self._contains(self.a_rect, x, y):
            return "A"
        if self._contains(self.b_rect, x, y):
            return "B"
        return None


class ContinuousCorridorAgent:
    """Minimal agent API needed by BucketStateEncoder."""

    def __init__(self, event_detector):
        transitions = {
            ("q0", None): ("q0", 0),
            ("q0", "A"): ("q1", 0),
            ("q1", None): ("q1", 0),
            ("q1", "B"): ("q2", 1),
            ("q2", None): ("q2", 0),
        }
        self._reward_machine = RewardMachine(transitions, event_detector)

    def get_reward_machine(self):
        return self._reward_machine


class ContinuousCorridorSequence:
    """A 2D continuous-state NMRDP with a sequential A-then-B reward."""

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
    ):
        self.rng = np.random.default_rng(seed)
        self.event_detector = event_detector or CorridorEventDetector()
        self.step_size = step_size
        self.noise_std = noise_std
        self.horizon = horizon
        self.x = 0.5
        self.y = 0.5
        self.q_idx = 0
        self.steps = 0

    def reset(self):
        self.x = float(self.rng.uniform(0.45, 0.55))
        self.y = float(self.rng.uniform(0.42, 0.58))
        self.q_idx = 0
        self.steps = 0
        return {"x": self.x, "y": self.y}, self.q_idx

    def _rm_transition(self, q_idx, event):
        if q_idx == 0 and event == "A":
            return 1, 0.0, False
        if q_idx == 1 and event == "B":
            return 2, 1.0, True
        if q_idx == 2:
            return 2, 0.0, True
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
        self.x = min(0.999999, max(0.0, self.x + dx * self.step_size + noise_x))
        self.y = min(0.999999, max(0.0, self.y + dy * self.step_size + noise_y))
        event = self.event_detector.detect_event({"x": self.x, "y": self.y})
        self.q_idx, reward, solved = self._rm_transition(self.q_idx, event)
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


def make_encoder(
    *,
    include_event_label: bool,
    buckets_x: int = 10,
    buckets_y: int = 4,
    event_detector: Optional[CorridorEventDetector] = None,
):
    detector = event_detector or CorridorEventDetector()
    return BucketStateEncoder(
        ContinuousCorridorAgent(detector),
        features=["x", "y"],
        lows=[0.0, 0.0],
        highs=[1.0, 1.0],
        buckets=[buckets_x, buckets_y],
        include_event_label=include_event_label,
        event_detector=detector,
        event_labels=["A", "B"],
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
    if algorithm == "QL":
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
) -> BucketQRMaxTrialResult:
    event_detector = CorridorEventDetector()
    encoder = make_encoder(
        include_event_label=include_event_label,
        buckets_x=buckets_x,
        buckets_y=buckets_y,
        event_detector=event_detector,
    )
    env = ContinuousCorridorSequence(
        seed=seed,
        event_detector=event_detector,
        step_size=step_size,
        noise_std=noise_std,
        horizon=horizon,
    )
    algo = _make_algorithm(algorithm, encoder, threshold, seed)

    first_perfect_eval_episode = None
    for episode in range(train_episodes):
        state, q_idx = env.reset()
        algo.learn_init_episode()
        for _ in range(horizon):
            encoded_state, info = encoder.encode(state, _rm_state_name(q_idx))
            action = algo.choose_action(encoded_state, best=False, info=info)
            next_state, next_q_idx, reward, stopped, solved = env.step(action)
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
                "qrm_experience": [],
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
    parser.add_argument("--algorithm", choices=["QRMAX", "QL", "RMAX"], default="QRMAX")
    parser.add_argument("--seeds", default="1700:1705")
    parser.add_argument("--buckets-x", type=int, default=10)
    parser.add_argument("--buckets-y", type=int, default=4)
    parser.add_argument("--threshold", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=0.003)
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
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
    )
    avg_len = summary["average_length_mean"]
    avg_len_text = "NA" if avg_len is None else f"{avg_len:.3f}"
    print(
        "algorithm,event_aware,buckets_x,buckets_y,threshold,noise,seeds,"
        "success_mean,success_min,success_max,avg_len"
    )
    print(
        f"{summary['algorithm']},{summary['include_event_label']},"
        f"{args.buckets_x},{args.buckets_y},{args.threshold},{args.noise_std},"
        f"{summary['num_seeds']},{summary['success_mean']:.3f},"
        f"{summary['success_min']:.3f},{summary['success_max']:.3f},"
        f"{avg_len_text}"
    )


if __name__ == "__main__":
    main()
