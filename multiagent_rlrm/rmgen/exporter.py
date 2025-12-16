import json
from pathlib import Path
from typing import Optional

from multiagent_rlrm.multi_agent.event_detector import EventDetector
from multiagent_rlrm.multi_agent.reward_machine import RewardMachine
from multiagent_rlrm.rmgen.spec import RMSpec


class PassthroughEventDetector(EventDetector):
    """
    Minimal event detector that simply returns the 'event' field from a state dict.
    Useful for generic/offline RM execution and tests.
    """

    def __init__(self, allowed_events):
        self.allowed_events = set(allowed_events)

    def detect_event(self, current_state):
        event = None
        if isinstance(current_state, dict):
            event = current_state.get("event")
        if event not in self.allowed_events:
            return None
        return event


def export_spec_to_file(spec: RMSpec, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(spec.to_dict(), f, indent=2, ensure_ascii=True)


def build_reward_machine(
    spec: RMSpec, event_detector: Optional[EventDetector] = None
) -> RewardMachine:
    """
    Instantiate a RewardMachine from an RMSpec. If no detector is provided,
    uses a passthrough detector keyed on the vocabulary.
    """
    detector = event_detector or PassthroughEventDetector(spec.event_vocabulary)
    transitions = spec.as_transition_map()
    rm = RewardMachine(transitions, detector)

    # Align RM current/initial state with spec.
    rm.initial_state = spec.initial_state
    rm.current_state = spec.initial_state
    rm.state_indices = rm._generate_state_indices()
    return rm
