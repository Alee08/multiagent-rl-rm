from typing import Dict, Iterable, Set

from multiagent_rlrm.rmgen.spec import RMSpec


class ValidationError(Exception):
    """Raised when an RMSpec is invalid."""


def _check_unique(items: Iterable[str], label: str) -> None:
    seen: Set[str] = set()
    for item in items:
        if item in seen:
            raise ValidationError(f"Duplicate {label}: {item}")
        seen.add(item)


def validate_schema(spec: RMSpec) -> None:
    if not spec.states:
        raise ValidationError("states must be non-empty")
    _check_unique(spec.states, "state")

    if spec.initial_state not in spec.states:
        raise ValidationError(f"initial_state {spec.initial_state} not in states")
    for t_state in spec.terminal_states:
        if t_state not in spec.states:
            raise ValidationError(f"terminal_state {t_state} not in states")
    _check_unique(spec.terminal_states, "terminal_state")

    if not spec.event_vocabulary:
        raise ValidationError("event_vocabulary must be non-empty")
    _check_unique(spec.event_vocabulary, "event")

    if not spec.transitions:
        raise ValidationError("transitions must be non-empty")

    state_set = set(spec.states)
    event_set = set(spec.event_vocabulary)
    for t in spec.transitions:
        if t.from_state not in state_set:
            raise ValidationError(f"transition from_state {t.from_state} not in states")
        if t.to_state not in state_set:
            raise ValidationError(f"transition to_state {t.to_state} not in states")
        if t.event not in event_set:
            raise ValidationError(f"transition event {t.event} not in vocabulary")

    # Ensure initial state is covered by at least one transition (source or sink)
    participates = any(
        t.from_state == spec.initial_state or t.to_state == spec.initial_state
        for t in spec.transitions
    )
    if not participates:
        raise ValidationError(
            f"initial_state {spec.initial_state} has no incident transitions"
        )


def ensure_deterministic(spec: RMSpec) -> None:
    seen: Dict[tuple, str] = {}
    for t in spec.transitions:
        key = (t.from_state, t.event)
        if key in seen and seen[key] != t.to_state:
            raise ValidationError(
                f"Non-deterministic transitions for {key}: {seen[key]} vs {t.to_state}"
            )
        seen[key] = t.to_state


def validate_spec(spec: RMSpec) -> None:
    """
    Runs all validations; raises ValidationError if any check fails.
    """
    validate_schema(spec)
    ensure_deterministic(spec)


def validate_semantics(
    spec: RMSpec,
    *,
    max_positive_reward_transitions: int = None,
    terminal_reward_must_be_zero: bool = True,
) -> None:
    """
    Optional semantic validations.
    """
    if max_positive_reward_transitions is not None:
        positive = [t for t in spec.transitions if t.reward > 0]
        if len(positive) > max_positive_reward_transitions:
            raise ValidationError(
                f"Positive-reward transitions exceed limit {max_positive_reward_transitions}: {positive}"
            )

    if terminal_reward_must_be_zero:
        terminal_set = set(spec.terminal_states)
        bad = [
            t
            for t in spec.transitions
            if t.from_state in terminal_set and t.reward != 0
        ]
        if bad:
            raise ValidationError(
                f"Terminal transitions must have reward 0. Offenders: {bad}"
            )
