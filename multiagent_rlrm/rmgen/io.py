from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from multiagent_rlrm.multi_agent.event_detector import EventDetector
from multiagent_rlrm.multi_agent.reward_machine import RewardMachine
from multiagent_rlrm.rmgen.completion import (
    complete_missing_transitions as complete_spec,
)
from multiagent_rlrm.rmgen.exporter import PassthroughEventDetector
from multiagent_rlrm.rmgen.spec import RMSpec
from multiagent_rlrm.rmgen.validator import validate_semantics, validate_spec


def _parse_json(text: str, *, source: Path) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {source}: {exc}") from exc


def _parse_yaml(text: str, *, source: Path) -> Any:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise ValueError(
            f"YAML support requires PyYAML. Install it with `pip install pyyaml`, "
            f"or provide a JSON spec instead: {source}"
        ) from exc

    try:
        return yaml.safe_load(text)
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise ValueError(f"Invalid YAML in {source}: {exc}") from exc


def load_rmspec(path: Union[str, Path]) -> RMSpec:
    """
    Load an RMSpec from a JSON or YAML file.

    YAML is optional: if PyYAML is not installed and the file is not valid JSON,
    this function raises a ValueError with installation instructions.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"RM spec file not found: {source}")
    if not source.is_file():
        raise ValueError(f"RM spec path is not a file: {source}")

    text = source.read_text(encoding="utf-8")
    suffix = source.suffix.lower()

    data: Any
    if suffix == ".json":
        data = _parse_json(text, source=source)
    elif suffix in {".yaml", ".yml"}:
        try:
            data = _parse_json(text, source=source)
        except ValueError:
            data = _parse_yaml(text, source=source)
    else:
        try:
            data = _parse_json(text, source=source)
        except ValueError:
            data = _parse_yaml(text, source=source)

    if not isinstance(data, dict):
        raise ValueError(f"RM spec must be a JSON/YAML object at top-level: {source}")

    try:
        return RMSpec.from_dict(data)
    except KeyError as exc:
        raise ValueError(f"RM spec missing required field {exc!s}: {source}") from exc
    except ValueError as exc:
        raise ValueError(f"RM spec has invalid values: {source}: {exc}") from exc


def _compile_transition_map(
    spec: RMSpec,
    *,
    event_mapping: Optional[Mapping[str, object]] = None,
) -> Dict[Tuple[object, object], Tuple[object, object]]:
    if not event_mapping:
        return spec.as_transition_map()

    transition_map: Dict[Tuple[object, object], Tuple[object, object]] = {}
    for t in spec.transitions:
        if t.event not in event_mapping:
            raise ValueError(
                f"Unknown event '{t.event}' in RMSpec; missing from event mapping "
                f"(available: {sorted(event_mapping.keys())})"
            )

        mapped = event_mapping[t.event]
        if mapped is None:
            raise ValueError(f"Event mapping for '{t.event}' is None")

        mapped_events = (
            list(mapped) if isinstance(mapped, (list, set, frozenset)) else [mapped]
        )
        if not mapped_events:
            raise ValueError(f"Event mapping for '{t.event}' is empty")

        for ev in mapped_events:
            try:
                hash(ev)
            except TypeError as exc:
                raise ValueError(
                    f"Mapped event for '{t.event}' is not hashable: {ev!r}"
                ) from exc

            key = (t.from_state, ev)
            value = (t.to_state, t.reward)
            if key in transition_map and transition_map[key] != value:
                raise ValueError(
                    f"Event mapping produced conflicting transitions for {key}: "
                    f"{transition_map[key]} vs {value}"
                )
            transition_map[key] = value

    return transition_map


def compile_reward_machine(
    spec: RMSpec,
    *,
    event_detector: Optional[EventDetector] = None,
    event_mapping: Optional[Mapping[str, object]] = None,
    complete_missing_transitions: bool = False,
    default_reward: float = 0.0,
    terminal_self_loop: bool = True,
    max_positive_reward_transitions: int = None,
    terminal_reward_must_be_zero: bool = True,
) -> RewardMachine:
    """
    Validate an RMSpec and compile it into a RewardMachine.

    Args:
        spec: The loaded RMSpec.
        event_detector: Optional EventDetector. If omitted, uses a passthrough
            detector that expects dict states with an "event" key.
        event_mapping: Optional mapping from spec event strings to environment
            event objects (e.g., grid positions). Values may also be lists/sets to
            expand one spec event into multiple environment events.
        complete_missing_transitions: If True, completes missing transitions with
            self-loops (same behavior/params as `multiagent_rlrm.cli.rmgen`).
    """
    if complete_missing_transitions:
        spec, _report = complete_spec(
            spec,
            default_reward=default_reward,
            terminal_self_loop=terminal_self_loop,
        )

    validate_spec(spec)
    validate_semantics(
        spec,
        max_positive_reward_transitions=max_positive_reward_transitions,
        terminal_reward_must_be_zero=terminal_reward_must_be_zero,
    )

    detector = event_detector or PassthroughEventDetector(spec.event_vocabulary)
    transitions = _compile_transition_map(spec, event_mapping=event_mapping)

    rm = RewardMachine(transitions, detector)
    rm.initial_state = spec.initial_state
    rm.current_state = spec.initial_state
    rm.state_indices = rm._generate_state_indices()
    return rm
