from multiagent_rlrm.environments.continuous_frozen_lake.bucket_qrmax_experiment import (
    FrozenLakeEventDetector,
    make_encoder,
    run_bucket_qrmax_sweep,
)
from multiagent_rlrm.environments.frozen_lake.config_frozen_lake import (
    config as frozenlake_config,
)
from multiagent_rlrm.utils.utils import parse_map_emoji


def test_continuous_frozen_lake_event_detector_finds_goals_and_holes():
    holes, goals, dimensions = parse_map_emoji(
        frozenlake_config["maps"]["map1"]["layout"]
    )
    detector = FrozenLakeEventDetector(
        goals=goals,
        holes=holes,
        width=dimensions[0],
        height=dimensions[1],
    )

    assert (
        detector.detect_event({"x": goals["A"][0] + 0.5, "y": goals["A"][1] + 0.5})
        == "A"
    )
    assert (
        detector.detect_event({"x": holes[0][0] + 0.5, "y": holes[0][1] + 0.5})
        == "hole"
    )
    assert detector.detect_event({"x": 1.5, "y": 1.5}) is None


def test_continuous_frozen_lake_encoder_is_finite_and_event_aware():
    encoder = make_encoder(
        include_event_label=True,
        bucket_mode="uniform",
        buckets_x=10,
        buckets_y=10,
    )
    state = {"x": 4.5, "y": 4.5}
    encoded, info = encoder.encode(state, "q0")

    assert 0 <= encoded < encoder.total_state_space_size
    assert info["event"] == "A"
    assert encoder.num_abstract_states == 10 * 10 * 5


def test_event_aware_bucket_qrmax_runs_on_continuous_frozen_lake_smoke():
    summary = run_bucket_qrmax_sweep(
        seeds=[2000],
        algorithm="QRMAX",
        include_event_label=True,
        bucket_mode="uniform",
        train_episodes=20,
        eval_episodes=5,
        horizon=80,
        threshold=2,
        noise_std=0.0,
        slip_probability=0.0,
    )

    assert summary["num_seeds"] == 1
    assert 0.0 <= summary["success_mean"] <= 1.0
