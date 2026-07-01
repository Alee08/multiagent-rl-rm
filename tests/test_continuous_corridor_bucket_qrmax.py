from multiagent_rlrm.environments.continuous_corridor.bucket_qrmax_experiment import (
    ContinuousCorridorSequence,
    CorridorEventDetector,
    LAYOUT_OBSTACLES,
    event_aligned_bucket_edges,
    make_encoder,
    run_bucket_qrmax_sweep,
    transition_probed_bucket_edges,
)


def test_continuous_corridor_event_sequence_reward():
    detector = CorridorEventDetector()
    env = ContinuousCorridorSequence(seed=1, event_detector=detector, noise_std=0.0)

    env.x, env.y = 0.38, 0.50
    _, q_idx, reward, stopped, solved = env.step(0)
    assert q_idx == 1
    assert reward == 0.0
    assert not stopped
    assert not solved

    env.x, env.y = 0.58, 0.50
    _, q_idx, reward, stopped, solved = env.step(1)
    assert q_idx == 2
    assert reward == 1.0
    assert stopped
    assert solved


def test_continuous_corridor_abc_sequence_reward():
    detector = CorridorEventDetector(event_labels=("A", "B", "C"))
    env = ContinuousCorridorSequence(
        seed=1,
        event_detector=detector,
        event_sequence=("A", "B", "C"),
        noise_std=0.0,
    )

    env.x, env.y = 0.38, 0.50
    _, q_idx, reward, stopped, solved = env.step(0)
    assert q_idx == 1
    assert reward == 0.0
    assert not stopped
    assert not solved

    env.x, env.y = 0.58, 0.50
    _, q_idx, reward, stopped, solved = env.step(1)
    assert q_idx == 2
    assert reward == 0.0
    assert not stopped
    assert not solved

    env.x, env.y = 0.78, 0.50
    _, q_idx, reward, stopped, solved = env.step(1)
    assert q_idx == 3
    assert reward == 1.0
    assert stopped
    assert solved


def test_continuous_corridor_bottleneck_blocks_wall_outside_gap():
    env = ContinuousCorridorSequence(
        seed=3,
        step_size=0.025,
        noise_std=0.0,
        obstacle_rects=LAYOUT_OBSTACLES["bottleneck"],
    )

    env.x, env.y = 0.46, 0.20
    state, _, _, _, _ = env.step(1)
    assert state["x"] == 0.46
    assert state["y"] == 0.20

    env.x, env.y = 0.46, 0.50
    state, _, _, _, _ = env.step(1)
    assert abs(state["x"] - 0.485) < 1e-12
    assert state["y"] == 0.50


def test_continuous_corridor_event_aware_encoder_splits_same_bucket():
    detector = CorridorEventDetector()
    encoder = make_encoder(
        include_event_label=True,
        buckets_x=10,
        buckets_y=4,
        event_detector=detector,
    )

    event_state = {"x": 0.36, "y": 0.50}
    nearby_nonevent_state = {"x": 0.32, "y": 0.50}
    assert encoder.bucket_tuple(event_state) == encoder.bucket_tuple(
        nearby_nonevent_state
    )
    assert encoder.abstract_state_index(event_state) != encoder.abstract_state_index(
        nearby_nonevent_state
    )


def test_continuous_corridor_event_aligned_edges_split_event_boundaries():
    detector = CorridorEventDetector(event_labels=("A", "B", "C"))
    x_edges, y_edges = event_aligned_bucket_edges(detector, 10, 4)

    for edge in (0.35, 0.45, 0.55, 0.65, 0.75, 0.85):
        assert edge in x_edges
    assert 0.30 in y_edges
    assert 0.70 in y_edges

    encoder = make_encoder(
        include_event_label=True,
        buckets_x=10,
        buckets_y=4,
        event_detector=detector,
        event_sequence=("A", "B", "C"),
        bucket_mode="event-aligned",
    )
    assert encoder.bucket_tuple({"x": 0.34, "y": 0.50}) != encoder.bucket_tuple(
        {"x": 0.36, "y": 0.50}
    )


def test_continuous_corridor_transition_probed_edges_find_bottleneck():
    detector = CorridorEventDetector(event_labels=("A", "B"))
    x_edges, y_edges = transition_probed_bucket_edges(
        detector,
        LAYOUT_OBSTACLES["bottleneck"],
        ("A", "B"),
        12,
        8,
    )

    assert 0.475 in x_edges
    assert 0.525 in x_edges
    assert 0.4 in y_edges
    assert 0.6 in y_edges


def test_continuous_corridor_reset_ranges_are_configurable():
    env = ContinuousCorridorSequence(
        seed=2,
        reset_x_range=(0.48, 0.49),
        reset_y_range=(0.10, 0.90),
    )

    for _ in range(20):
        state, q_idx = env.reset()
        assert 0.48 <= state["x"] <= 0.49
        assert 0.10 <= state["y"] <= 0.90
        assert q_idx == 0


def test_event_aware_bucket_qrmax_solves_continuous_corridor_task():
    summary = run_bucket_qrmax_sweep(
        seeds=range(1700, 1704),
        include_event_label=True,
        train_episodes=150,
        eval_episodes=50,
        threshold=8,
        buckets_x=10,
        buckets_y=4,
    )

    assert summary["success_min"] == 1.0
    assert summary["average_length_mean"] < 15
    assert summary["training_successes_mean"] > 0
    assert set(summary["event_count_means"]) == {"A", "B"}


def test_qrm_baseline_runs_on_continuous_corridor_task():
    summary = run_bucket_qrmax_sweep(
        seeds=range(1700, 1702),
        algorithm="QRM",
        include_event_label=True,
        train_episodes=80,
        eval_episodes=20,
        threshold=8,
        buckets_x=10,
        buckets_y=4,
    )

    assert summary["success_min"] == 1.0
    assert summary["event_count_means"]["A"] > 0
    assert summary["event_count_means"]["B"] > 0


def test_event_label_ablation_changes_continuous_corridor_performance():
    summary = run_bucket_qrmax_sweep(
        seeds=range(1700, 1704),
        include_event_label=False,
        train_episodes=150,
        eval_episodes=50,
        threshold=8,
        buckets_x=10,
        buckets_y=4,
    )

    assert summary["success_mean"] <= 0.25
