from multiagent_rlrm.environments.continuous_line.bucket_qrmax_experiment import (
    IntervalEventDetector,
    make_encoder,
    run_bucket_qrmax_sweep,
)


def test_event_aware_bucket_qrmax_solves_continuous_line_task():
    summary = run_bucket_qrmax_sweep(
        seeds=range(1300, 1305),
        include_event_label=True,
        train_episodes=150,
        eval_episodes=50,
        threshold=8,
    )

    assert summary["success_min"] == 1.0
    assert summary["average_length_mean"] < 10


def test_event_aware_encoder_splits_event_region_from_same_bucket():
    detector = IntervalEventDetector()
    encoder = make_encoder(
        include_event_label=True,
        buckets=10,
        event_detector=detector,
    )

    event_state = {"x": 0.36}
    nearby_nonevent_state = {"x": 0.32}
    assert encoder.bucket_tuple(event_state) == encoder.bucket_tuple(
        nearby_nonevent_state
    )
    assert encoder.abstract_state_index(event_state) != encoder.abstract_state_index(
        nearby_nonevent_state
    )
