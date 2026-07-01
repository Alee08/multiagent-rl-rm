# Continuous Line Bucket QR-MAX

This package contains a small continuous-state NMRDP used to validate
bucketized tabular learning with Reward Machines.

The environment is one-dimensional. The agent starts near the center and has
two discrete actions: move left or move right. The Reward Machine gives reward
after the sequence `A -> B`, where `A` and `B` are detected in two continuous
intervals.

The experiment is intentionally small. Its purpose is to check QR-MAX on a
finite abstraction of a continuous state space when the abstraction preserves
states that trigger different RM events.

Run the event-aware Bucket QR-MAX sanity check:

```bash
python -m multiagent_rlrm.environments.continuous_line.bucket_qrmax_experiment \
  --algorithm QRMAX \
  --event-aware \
  --buckets 10 \
  --threshold 8 \
  --seeds 1500:1520
```

Expected pattern:

- event-aware buckets solve the task with the same bucket count;
- sufficiently fine buckets may also work empirically, but event labels provide
  a direct consistency check for RM events.
