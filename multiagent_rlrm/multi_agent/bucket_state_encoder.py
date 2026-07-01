from bisect import bisect_right
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
        bucket_edges: Optional[Sequence[Sequence[float]]] = None,
    ):
        super().__init__(agent)
        if not (len(features) == len(lows) == len(highs) == len(buckets)):
            raise ValueError(
                "features, lows, highs, and buckets must have the same length"
            )
        if bucket_edges is not None and len(bucket_edges) != len(features):
            raise ValueError("bucket_edges must have one edge sequence per feature")
        if not features:
            raise ValueError("at least one continuous feature is required")

        self.features = tuple(features)
        self.lows = tuple(float(v) for v in lows)
        self.highs = tuple(float(v) for v in highs)
        self.include_event_label = include_event_label
        self.event_detector = event_detector
        self.feature_extractor = feature_extractor

        for low, high, bucket_count in zip(
            self.lows, self.highs, tuple(int(v) for v in buckets)
        ):
            if high <= low:
                raise ValueError("each high bound must be greater than its low bound")
            if bucket_count <= 0:
                raise ValueError("bucket counts must be positive")

        if bucket_edges is None:
            self.bucket_edges = None
            self.buckets = tuple(int(v) for v in buckets)
        else:
            normalized_edges = []
            for feature_edges, low, high in zip(
                bucket_edges, self.lows, self.highs
            ):
                edges = tuple(float(edge) for edge in feature_edges)
                if len(edges) < 2:
                    raise ValueError("each bucket_edges entry needs at least 2 edges")
                if any(right <= left for left, right in zip(edges, edges[1:])):
                    raise ValueError("bucket_edges entries must be strictly increasing")
                if edges[0] != low or edges[-1] != high:
                    raise ValueError(
                        "bucket_edges entries must start at low and end at high"
                    )
                normalized_edges.append(edges)
            self.bucket_edges = tuple(normalized_edges)
            self.buckets = tuple(len(edges) - 1 for edges in self.bucket_edges)

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
            if self.bucket_edges is None:
                width = (high - low) / bucket_count
                bucket_ids.append(
                    min(bucket_count - 1, int(floor((value - low) / width)))
                )
            else:
                edges = self.bucket_edges[len(bucket_ids)]
                bucket_ids.append(
                    min(bucket_count - 1, max(0, bisect_right(edges, value) - 1))
                )
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
