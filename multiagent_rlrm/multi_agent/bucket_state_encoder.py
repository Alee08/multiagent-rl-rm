from math import floor
from typing import Callable, Mapping, Optional, Sequence

from multiagent_rlrm.multi_agent.state_encoder import StateEncoder


class BucketStateEncoder(StateEncoder):
    """Encode continuous observations into finite buckets for tabular algorithms.

    When ``include_event_label`` is enabled, the detected Reward Machine event is
    appended to the bucket index. This prevents a single bucket from mixing
    states that trigger different RM transitions.
    """

    def __init__(
        self,
        agent,
        features: Sequence[str],
        lows: Sequence[float],
        highs: Sequence[float],
        buckets: Sequence[int],
        *,
        include_event_label: bool = False,
        event_detector=None,
        event_labels: Optional[Sequence[object]] = None,
        feature_extractor: Optional[Callable[[Mapping[str, float], str], float]] = None,
    ):
        super().__init__(agent)
        if not (len(features) == len(lows) == len(highs) == len(buckets)):
            raise ValueError(
                "features, lows, highs, and buckets must have the same length"
            )
        if not features:
            raise ValueError("at least one continuous feature is required")

        self.features = tuple(features)
        self.lows = tuple(float(v) for v in lows)
        self.highs = tuple(float(v) for v in highs)
        self.buckets = tuple(int(v) for v in buckets)
        self.include_event_label = include_event_label
        self.event_detector = event_detector
        self.feature_extractor = feature_extractor

        for low, high, bucket_count in zip(self.lows, self.highs, self.buckets):
            if high <= low:
                raise ValueError("each high bound must be greater than its low bound")
            if bucket_count <= 0:
                raise ValueError("bucket counts must be positive")

        self._bucket_multipliers = []
        multiplier = 1
        for bucket_count in reversed(self.buckets):
            self._bucket_multipliers.insert(0, multiplier)
            multiplier *= bucket_count
        self.num_bucket_states = multiplier

        if include_event_label:
            if event_detector is None:
                raise ValueError(
                    "event_detector is required when include_event_label=True"
                )
            labels = [None]
            if event_labels is not None:
                labels.extend(label for label in event_labels if label is not None)
            self.event_to_index = {label: idx for idx, label in enumerate(labels)}
        else:
            self.event_to_index = {None: 0}

    @property
    def num_event_labels(self):
        return len(self.event_to_index)

    @property
    def num_abstract_states(self):
        return self.num_bucket_states * self.num_event_labels

    @property
    def total_state_space_size(self):
        return self.num_abstract_states * self.agent.get_reward_machine().numbers_state()

    def _feature_value(self, state, feature):
        if self.feature_extractor is not None:
            return float(self.feature_extractor(state, feature))
        return float(state[feature])

    def bucket_tuple(self, state):
        bucket_ids = []
        for feature, low, high, bucket_count in zip(
            self.features, self.lows, self.highs, self.buckets
        ):
            value = self._feature_value(state, feature)
            if value <= low:
                bucket_ids.append(0)
                continue
            if value >= high:
                bucket_ids.append(bucket_count - 1)
                continue
            width = (high - low) / bucket_count
            bucket_ids.append(min(bucket_count - 1, int(floor((value - low) / width))))
        return tuple(bucket_ids)

    def bucket_index(self, state):
        return sum(
            bucket_id * multiplier
            for bucket_id, multiplier in zip(
                self.bucket_tuple(state), self._bucket_multipliers
            )
        )

    def detect_event(self, state):
        if not self.include_event_label:
            return None
        event = self.event_detector.detect_event(state)
        if event not in self.event_to_index:
            raise ValueError(
                f"Unknown event label {event!r}; pass it in event_labels when creating "
                "BucketStateEncoder."
            )
        return event

    def abstract_state_index(self, state):
        bucket_idx = self.bucket_index(state)
        event = self.detect_event(state)
        event_idx = self.event_to_index[event]
        return bucket_idx * self.num_event_labels + event_idx

    def encode(self, state, state_rm=None):
        rm_state_index = self.encode_rm_state(state_rm)
        abstract_idx = self.abstract_state_index(state)
        encoded_state = (
            abstract_idx * self.agent.get_reward_machine().numbers_state()
            + rm_state_index
        )
        info = {
            "s": abstract_idx,
            "q": rm_state_index,
            "bucket": self.bucket_tuple(state),
            "event": self.detect_event(state),
        }
        return encoded_state, info
