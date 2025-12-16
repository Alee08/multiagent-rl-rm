from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Dict, Mapping, Optional, Sequence, Set, Union

from multiagent_rlrm.rmgen.spec import RMSpec, TransitionSpec
from multiagent_rlrm.rmgen.validator import ValidationError


def normalize_event_key(raw: str) -> str:
    """
    Normalization key for matching event tokens:
    - strips whitespace
    - removes inner spaces (e.g., "At( A )" -> "at(a)")
    - lowercases everything
    """
    if raw is None:
        return ""
    # Remove ALL whitespace, not just at the ends.
    compact = "".join(str(raw).split())
    return compact.lower()


def enforce_env_id(
    spec: RMSpec,
    expected_env_id: str,
    *,
    reason: str,
) -> RMSpec:
    """
    Force env_id to an expected value and emit a warning when a mismatch is found.

    This is meant to be used as a guardrail in context-specific entrypoints
    (e.g., OfficeWorld), so a model/user typo like "officework" does not block runs.
    """
    expected = str(expected_env_id).strip().lower()
    current = (spec.env_id or "").strip().lower()
    if current != expected:
        old_display = spec.env_id if spec.env_id else "<missing>"
        print(
            f"Warning: overriding env_id from '{old_display}' to '{expected}' because {reason}"
        )
        spec.env_id = expected
    return spec


def autofix_rmspec_states_for_officeworld(spec: RMSpec) -> RMSpec:
    """
    OfficeWorld guardrail: auto-add missing states referenced by transitions.

    If a transition references a state not present in `spec.states`, the state is added.
    Additionally, if a missing state looks like a dedicated terminal state (heuristic:
    it appears only as a `to_state` of transitions with reward>0), it is also added to
    `terminal_states`.
    """
    state_set = set(spec.states)
    referenced = set()
    for t in spec.transitions:
        referenced.add(t.from_state)
        referenced.add(t.to_state)

    missing = sorted(s for s in referenced if s not in state_set)
    if not missing:
        return spec

    new_states = list(spec.states)
    new_terminal_states = list(spec.terminal_states)
    terminal_set = set(new_terminal_states)

    # Add missing states first.
    for s in missing:
        if s not in state_set:
            new_states.append(s)
            state_set.add(s)

    # Dedicated terminal heuristic for missing states:
    # - never appears as from_state
    # - appears as to_state with reward>0 at least once
    # - never appears as to_state with reward<=0
    for s in missing:
        appears_as_from = any(t.from_state == s for t in spec.transitions)
        appears_to_pos = any(t.to_state == s and t.reward > 0 for t in spec.transitions)
        appears_to_nonpos = any(
            t.to_state == s and t.reward <= 0 for t in spec.transitions
        )
        if (not appears_as_from) and appears_to_pos and (not appears_to_nonpos):
            if s not in terminal_set:
                new_terminal_states.append(s)
                terminal_set.add(s)

    spec.states = new_states
    spec.terminal_states = new_terminal_states
    return spec


def _next_q_state_name(existing_states: Set[str]) -> str:
    """
    Pick a fresh RM state name.

    Prefers q{N} style (q0, q1, ...) when possible to match common specs.
    """
    numbers = []
    for s in existing_states:
        m = re.fullmatch(r"q(\d+)", str(s))
        if m:
            numbers.append(int(m.group(1)))
    if numbers:
        candidate = f"q{max(numbers) + 1}"
        if candidate not in existing_states:
            return candidate

    base = "q_terminal"
    if base not in existing_states:
        return base
    i = 1
    while True:
        candidate = f"{base}_{i}"
        if candidate not in existing_states:
            return candidate
        i += 1


def autofix_terminal_reward_violations_for_officeworld(spec: RMSpec) -> RMSpec:
    """
    OfficeWorld guardrail: fix common LLM mistakes where a terminal state has
    a non-zero outgoing reward transition.

    This avoids failing `terminal_reward_must_be_zero=True` for specs that
    accidentally put the final reward on a terminal self-loop (e.g., q3 --e--> q3
    with reward=1). The fix is:
      - remove offending `from_state` from terminal_states
      - if the offender is a self-loop, redirect it into a fresh terminal sink state
        so the non-zero reward is only obtained once.
    """
    terminal_set = set(spec.terminal_states)
    offenders = [
        t for t in spec.transitions if t.from_state in terminal_set and t.reward != 0
    ]
    if not offenders:
        return spec

    original_terminal_set = set(spec.terminal_states)
    state_set = set(spec.states)

    remove_terminals: Set[str] = set()
    added_terminal_states: Set[str] = set()
    redirected_terminal: Dict[str, str] = {}

    new_states = list(spec.states)
    new_transitions = []

    for t in spec.transitions:
        if t.from_state not in original_terminal_set or t.reward == 0:
            new_transitions.append(t)
            continue

        remove_terminals.add(t.from_state)

        # If the model put the reward on a terminal self-loop, redirect it to a fresh
        # terminal sink state to avoid repeated rewards.
        if t.to_state == t.from_state:
            terminal_sink = redirected_terminal.get(t.from_state)
            if terminal_sink is None:
                terminal_sink = _next_q_state_name(state_set)
                state_set.add(terminal_sink)
                new_states.append(terminal_sink)
                added_terminal_states.add(terminal_sink)
                redirected_terminal[t.from_state] = terminal_sink
                print(
                    "Warning: rewrote a terminal self-loop with non-zero reward "
                    f"({t.from_state} --{t.event}--> {t.to_state}, reward={t.reward}) "
                    f"to transition into a new terminal state '{terminal_sink}'"
                )

            new_transitions.append(
                TransitionSpec(
                    from_state=t.from_state,
                    event=t.event,
                    to_state=terminal_sink,
                    reward=t.reward,
                )
            )
            continue

        # Non-self-loop case: treat the terminal label as a mistake and keep the
        # transition unchanged (the validator will stop considering it 'terminal').
        print(
            "Warning: removing state "
            f"'{t.from_state}' from terminal_states because it has a non-zero "
            f"outgoing reward transition (reward={t.reward})"
        )
        new_transitions.append(t)

    new_terminal_states = [s for s in spec.terminal_states if s not in remove_terminals]
    terminal_state_set = set(new_terminal_states)
    for s in sorted(added_terminal_states):
        if s not in terminal_state_set:
            new_terminal_states.append(s)
            terminal_state_set.add(s)

    spec.states = new_states
    spec.terminal_states = new_terminal_states
    spec.transitions = new_transitions
    return spec


class UnknownEventError(ValidationError):
    """
    Raised when an RMSpec event cannot be mapped to an allowed/canonical event.
    """

    def __init__(
        self,
        *,
        event: str,
        allowed_events: Sequence[str],
        map_name: Optional[str] = None,
        hint: Optional[str] = None,
    ):
        self.event = event
        self.allowed_events = list(allowed_events)
        self.map_name = map_name
        self.hint = hint
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        limit = 25
        allowed_sorted = sorted(self.allowed_events)
        shown = allowed_sorted[:limit]
        tail = "" if len(allowed_sorted) <= limit else ", ..."
        map_part = f" (map={self.map_name})" if self.map_name else ""
        hint_part = f" Hint: {self.hint}" if self.hint else ""
        return (
            f"Unknown event '{self.event}'{map_part}. "
            f"Allowed events: {', '.join(shown)}{tail}.{hint_part}"
        )


def _canonicalize_event(event: str, context: Mapping[str, object]) -> str:
    allowed_events = context.get("allowed_events")
    canonical_map = context.get("canonical_map")
    map_name = context.get("map_name")
    context_env_id = context.get("env_id")

    if not isinstance(allowed_events, Sequence) or not allowed_events:
        raise ValueError("context['allowed_events'] must be a non-empty sequence")
    if not isinstance(canonical_map, Mapping) or not canonical_map:
        raise ValueError("context['canonical_map'] must be a non-empty mapping")

    key = normalize_event_key(event)
    if key not in canonical_map:
        env_part = (
            str(context_env_id).strip().lower()
            if context_env_id is not None
            else "<context>"
        )
        map_part = str(map_name) if map_name else "<map>"
        raise UnknownEventError(
            event=event,
            allowed_events=list(allowed_events),
            map_name=str(map_name) if map_name else None,
            hint=f"Use `rmgen --context {env_part} --map {map_part}` to autoprompt allowed events.",
        )

    canonical = str(canonical_map[key])
    if canonical not in set(allowed_events):
        raise UnknownEventError(
            event=event,
            allowed_events=list(allowed_events),
            map_name=str(map_name) if map_name else None,
        )
    return canonical


def normalize_rmspec_events(
    rmspec: Union[Dict[str, Any], RMSpec],
    context: Mapping[str, object],
) -> Union[Dict[str, Any], RMSpec]:
    """
    Normalize RMSpec event strings using an environment-derived context.

    - Normalizes every entry in event_vocabulary
    - Normalizes every transition.event
    - Prefers canonical form 'at(X)' (but accepts bare tokens like 'A')
    - Raises UnknownEventError for unknown/unmappable events.
    """
    if isinstance(rmspec, dict):
        data: Dict[str, Any] = dict(rmspec)
        vocab = data.get("event_vocabulary", [])
        transitions = data.get("transitions", [])

        normalized_vocab = []
        seen_vocab = set()
        for ev in vocab:
            canon = _canonicalize_event(str(ev), context)
            if canon not in seen_vocab:
                seen_vocab.add(canon)
                normalized_vocab.append(canon)

        normalized_transitions = []
        for t in transitions:
            t2 = dict(t)
            canon = _canonicalize_event(str(t2.get("event", "")), context)
            t2["event"] = canon
            normalized_transitions.append(t2)

        # Ensure vocabulary covers all transition events (more robust for imperfect specs).
        for t in normalized_transitions:
            ev = t["event"]
            if ev not in seen_vocab:
                seen_vocab.add(ev)
                normalized_vocab.append(ev)

        data["event_vocabulary"] = normalized_vocab
        data["transitions"] = normalized_transitions
        return data

    if not isinstance(rmspec, RMSpec):
        raise TypeError(f"Unsupported RMSpec type: {type(rmspec)}")

    normalized_vocab = []
    seen_vocab = set()
    for ev in rmspec.event_vocabulary:
        canon = _canonicalize_event(ev, context)
        if canon not in seen_vocab:
            seen_vocab.add(canon)
            normalized_vocab.append(canon)

    normalized_transitions = []
    seen_transitions = set()
    for t in rmspec.transitions:
        canon = _canonicalize_event(t.event, context)
        t2 = TransitionSpec(
            from_state=t.from_state, event=canon, to_state=t.to_state, reward=t.reward
        )
        key = (t2.from_state, t2.event, t2.to_state, t2.reward)
        if key in seen_transitions:
            continue
        seen_transitions.add(key)
        normalized_transitions.append(t2)

    # Ensure vocabulary covers all transition events.
    for t in normalized_transitions:
        if t.event not in seen_vocab:
            seen_vocab.add(t.event)
            normalized_vocab.append(t.event)

    return replace(
        rmspec,
        event_vocabulary=normalized_vocab,
        transitions=normalized_transitions,
    )
