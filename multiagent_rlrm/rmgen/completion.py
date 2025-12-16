from typing import Dict, List, Tuple

from multiagent_rlrm.rmgen.spec import RMSpec, TransitionSpec


def complete_missing_transitions(
    spec: RMSpec,
    default_reward: float = 0.0,
    terminal_self_loop: bool = True,
) -> Tuple[RMSpec, Dict[str, int]]:
    """
    Complete missing transitions with self-loops and a default reward.

    Returns the updated spec and a report dict with counts.
    """
    existing = {(t.from_state, t.event): t for t in spec.transitions}
    added: List[TransitionSpec] = []

    if not spec.states or not spec.event_vocabulary:
        return spec, {"added": 0}

    for state in spec.states:
        for event in spec.event_vocabulary:
            key = (state, event)
            if key in existing:
                continue
            # For terminal states, we may opt out of adding self-loops.
            if state in spec.terminal_states and not terminal_self_loop:
                continue
            t = TransitionSpec(
                from_state=state, event=event, to_state=state, reward=default_reward
            )
            added.append(t)

    if added:
        spec.transitions.extend(added)

    return spec, {"added": len(added)}
