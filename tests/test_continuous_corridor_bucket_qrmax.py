from multiagent_rlrm.environments.continuous_corridor.bucket_qrmax_experiment import (
    ContinuousCorridorSequence,
    CorridorEventDetector,
    make_encoder,
    run_bucket_qrmax_sweep,
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


def test_plain_bucket_qrmax_is_not_robust_on_continuous_corridor_task():
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
