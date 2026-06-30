# Continuous Corridor Bucket QR-MAX

This package contains a controlled two-dimensional continuous-state NMRDP used
to stress Bucket QR-MAX beyond the one-dimensional continuous-line task.

The state is `(x, y) in [0, 1]^2`. The agent starts near the center of the
corridor and has four discrete actions: left, right, down, and up. The Reward
Machine gives reward after the ordered sequence `A -> B`, where `A` and `B` are
detected in continuous rectangular regions.

The task is intentionally still small. Its purpose is to check that QR-MAX can
handle a 2D finite abstraction when the abstraction preserves RM events. With
the default `10 x 4` buckets, event-aware buckets solve the task robustly, while
plain buckets are much less reliable because they mix event and non-event states.

Run the event-aware QR-MAX check:

```bash
python -m multiagent_rlrm.environments.continuous_corridor.bucket_qrmax_experiment \
  --algorithm QRMAX \
  --event-aware \
  --buckets-x 10 \
  --buckets-y 4 \
  --threshold 8 \
  --seeds 1700:1705
```

Run the corresponding plain bucket baseline:

```bash
python -m multiagent_rlrm.environments.continuous_corridor.bucket_qrmax_experiment \
  --algorithm QRMAX \
  --buckets-x 10 \
  --buckets-y 4 \
  --threshold 8 \
  --seeds 1700:1705
```
