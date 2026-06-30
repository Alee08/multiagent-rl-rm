from multiagent_rlrm.environments.continuous_line.bucket_qrmax_experiment import (
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


def test_plain_bucket_qrmax_fails_when_bucket_mixes_rm_events():
    summary = run_bucket_qrmax_sweep(
        seeds=range(1300, 1305),
        include_event_label=False,
        train_episodes=150,
        eval_episodes=50,
        threshold=8,
    )

    assert summary["success_max"] == 0.0
