import pytest

from multiagent_rlrm.learning_algorithms.qrmax_v2 import QRMax_v2
from multiagent_rlrm.multi_agent.bucket_state_encoder import BucketStateEncoder
from multiagent_rlrm.multi_agent.reward_machine import RewardMachine


class IntervalEventDetector:
    def detect_event(self, state):
        x = state["x"]
        if 0.35 <= x <= 0.45:
            return "A"
        if 0.55 <= x <= 0.65:
            return "B"
        return None


class DummyAgent:
    def __init__(self):
        transitions = {
            ("q0", None): ("q0", 0),
            ("q0", "A"): ("q1", 0),
            ("q1", None): ("q1", 0),
            ("q1", "B"): ("q2", 1),
            ("q2", None): ("q2", 0),
        }
        self._rm = RewardMachine(transitions, IntervalEventDetector())

    def get_reward_machine(self):
        return self._rm


def make_encoder(include_event_label):
    return BucketStateEncoder(
        DummyAgent(),
        features=["x"],
        lows=[0.0],
        highs=[1.0],
        buckets=[10],
        include_event_label=include_event_label,
        event_detector=IntervalEventDetector(),
        event_labels=["A", "B"],
    )


def test_bucket_state_encoder_splits_states_by_event_label():
    plain = make_encoder(include_event_label=False)
    event_aware = make_encoder(include_event_label=True)

    no_event_state = {"x": 0.32}
    event_state = {"x": 0.37}

    assert plain.bucket_tuple(no_event_state) == plain.bucket_tuple(event_state)
    assert plain.encode(no_event_state, "q0")[1]["s"] == plain.encode(
        event_state, "q0"
    )[1]["s"]

    no_event_info = event_aware.encode(no_event_state, "q0")[1]
    event_info = event_aware.encode(event_state, "q0")[1]
    assert no_event_info["bucket"] == event_info["bucket"]
    assert no_event_info["event"] is None
    assert event_info["event"] == "A"
    assert no_event_info["s"] != event_info["s"]


def test_bucket_state_encoder_rejects_unknown_event_labels():
    encoder = BucketStateEncoder(
        DummyAgent(),
        features=["x"],
        lows=[0.0],
        highs=[1.0],
        buckets=[10],
        include_event_label=True,
        event_detector=IntervalEventDetector(),
        event_labels=["A"],
    )

    with pytest.raises(ValueError, match="Unknown event label"):
        encoder.encode({"x": 0.60}, "q0")


def test_event_aware_buckets_preserve_qrmax_rm_transition_determinism():
    plain = make_encoder(include_event_label=False)
    event_aware = make_encoder(include_event_label=True)

    def apply_two_transitions(encoder):
        algo = QRMax_v2(
            state_space_size=encoder.total_state_space_size,
            action_space_size=1,
            q_space_size=encoder.agent.get_reward_machine().numbers_state(),
            gamma=0.9,
            nsamplesTE=10,
            nsamplesRE=10,
            nsamplesTQ=10,
            nsamplesRQ=10,
        )
        algo.learn_init()
        _, first_next = encoder.encode({"x": 0.32}, "q0")
        _, second_next = encoder.encode({"x": 0.37}, "q1")
        action = 0

        algo.update(
            0,
            0,
            action,
            0,
            False,
            info={
                "prev_s": 0,
                "prev_q": 0,
                "s": first_next["s"],
                "q": first_next["q"],
                "Renv": 0,
                "RQ": 0,
                "qrm_experience": [],
            },
        )
        algo.update(
            0,
            0,
            action,
            0,
            False,
            info={
                "prev_s": 0,
                "prev_q": 0,
                "s": second_next["s"],
                "q": second_next["q"],
                "Renv": 0,
                "RQ": 0,
                "qrm_experience": [],
            },
        )

    with pytest.raises(AssertionError):
        apply_two_transitions(plain)

    apply_two_transitions(event_aware)
