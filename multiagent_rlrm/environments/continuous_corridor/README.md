# Continuous Corridor Bucket QR-MAX

This package contains a controlled two-dimensional continuous-state NMRDP used
to stress Bucket QR-MAX beyond the one-dimensional continuous-line task.

The state is `(x, y) in [0, 1]^2`. The agent starts near the center of the
corridor and has four discrete actions: left, right, down, and up. The default
Reward Machine gives reward after the ordered sequence `A -> B`, where `A` and
`B` are detected in continuous rectangular regions.

The task is intentionally still small. Its purpose is to check that QR-MAX can
handle a 2D finite abstraction when the abstraction preserves RM events. With
the default `10 x 4` buckets, event-aware buckets solve the task robustly. The
non-event-aware setting is available as a controlled ablation of the event label.

The runner also exposes controlled stress cases:

- `--sequence ABC` extends the Reward Machine to `A -> B -> C`.
- `--layout bottleneck` inserts a vertical wall with a central passage.
- `--bucket-mode event-aligned` adds bucket edges at RM event boundaries.
- `--bucket-mode transition-probed` probes one-step transition changes and
  adds refined bucket edges at inferred dynamics boundaries.
- the reported `event_count_means` and `training_successes_mean` help diagnose
  whether performance differences come from event detection coverage or from
  planning on the learned abstract model.

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

Run the corresponding non-event-aware bucket comparison:

```bash
python -m multiagent_rlrm.environments.continuous_corridor.bucket_qrmax_experiment \
  --algorithm QRMAX \
  --buckets-x 10 \
  --buckets-y 4 \
  --threshold 8 \
  --seeds 1700:1705
```

Run the harder reset/noise check:

```bash
python -m multiagent_rlrm.environments.continuous_corridor.bucket_qrmax_experiment \
  --algorithm QRMAX \
  --event-aware \
  --bucket-mode event-aligned \
  --buckets-x 10 \
  --buckets-y 6 \
  --threshold 18 \
  --noise-std 0.006 \
  --reset-y-low 0.10 \
  --reset-y-high 0.90 \
  --train-episodes 400 \
  --seeds 1810:1820
```

Run the deterministic bottleneck sanity check:

```bash
python -m multiagent_rlrm.environments.continuous_corridor.bucket_qrmax_experiment \
  --algorithm QRMAX \
  --event-aware \
  --layout bottleneck \
  --buckets-x 10 \
  --buckets-y 4 \
  --threshold 8 \
  --noise-std 0.0 \
  --reset-x-low 0.40 \
  --reset-x-high 0.46 \
  --reset-y-low 0.42 \
  --reset-y-high 0.58 \
  --train-episodes 300 \
  --seeds 1850:1855
```

Run the `A -> B -> C` stress case with event-aligned buckets:

```bash
python -m multiagent_rlrm.environments.continuous_corridor.bucket_qrmax_experiment \
  --algorithm QRMAX \
  --event-aware \
  --bucket-mode event-aligned \
  --sequence ABC \
  --horizon 120 \
  --buckets-x 12 \
  --buckets-y 4 \
  --threshold 10 \
  --train-episodes 250 \
  --seeds 1900:1905
```

Run the noisy bottleneck check with transition-probed buckets. This mode does
not read the obstacle rectangles as bucket edges; it probes one-step transition
outcomes, refines recurring dynamics boundaries by bisection, and then trains
QR-MAX on the resulting finite abstraction.

```bash
python -m multiagent_rlrm.environments.continuous_corridor.bucket_qrmax_experiment \
  --algorithm QRMAX \
  --event-aware \
  --bucket-mode transition-probed \
  --layout bottleneck \
  --buckets-x 12 \
  --buckets-y 8 \
  --threshold 10 \
  --noise-std 0.003 \
  --reset-x-low 0.40 \
  --reset-x-high 0.46 \
  --reset-y-low 0.20 \
  --reset-y-high 0.80 \
  --horizon 120 \
  --train-episodes 800 \
  --eval-episodes 50 \
  --seeds 1850:1855
```
