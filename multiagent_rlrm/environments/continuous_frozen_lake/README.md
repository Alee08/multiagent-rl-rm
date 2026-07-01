# Continuous Frozen Lake Bucket QR-MAX

This experiment keeps the Frozen Lake map and Reward Machine structure while
representing the agent position as continuous `(x, y)` coordinates. Goal and
hole events are detected from the grid cell containing the continuous position.

The runner uses fixed buckets over the continuous map extent. On the default
`10 x 10` Frozen Lake map, the default bucket resolution matches the map's cell
scale while the underlying state remains continuous.

Example:

```bash
python -m multiagent_rlrm.environments.continuous_frozen_lake.bucket_qrmax_experiment \
  --algorithm QRMAX \
  --event-aware \
  --sequence ABC \
  --bucket-mode uniform \
  --buckets-x 10 \
  --buckets-y 10 \
  --threshold 8 \
  --seeds 2000:2005
```
