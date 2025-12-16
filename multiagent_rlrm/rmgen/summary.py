from __future__ import annotations

from typing import Optional, Sequence

from multiagent_rlrm.rmgen.spec import RMSpec


def format_rmspec_summary(
    spec: RMSpec,
    *,
    agent_names: Optional[Sequence[str]] = None,
    source: Optional[str] = None,
    max_core_transition_lines: int = 200,
) -> str:
    """
    Build a readable, stable summary for an RMSpec.
    """
    lines = []

    header = "Reward Machine summary"
    if source:
        header += f" (from {source})"
    lines.append(header)

    if agent_names:
        agent_names_list = list(agent_names)
        if len(agent_names_list) == 1:
            lines.append(f"Agent: {agent_names_list[0]}")
        else:
            joined = ", ".join(agent_names_list)
            lines.append(f"Shared by agents: [{joined}]")

    lines.append(f"name: {spec.name}")
    lines.append(f"env_id: {spec.env_id}")
    lines.append(
        f"states: {len(spec.states)} (initial: {spec.initial_state}, terminal: {spec.terminal_states})"
    )
    lines.append(
        f"event_vocabulary ({len(spec.event_vocabulary)}): {', '.join(spec.event_vocabulary)}"
    )
    lines.append(f"transitions_total: {len(spec.transitions)}")

    core = [t for t in spec.transitions if t.from_state != t.to_state]
    core_sorted = sorted(core, key=lambda t: (t.from_state, t.event, t.to_state))
    lines.append(f"core_transitions_count: {len(core_sorted)}")
    lines.append("core_transitions (excluding self-loops):")
    shown = (
        core_sorted
        if max_core_transition_lines is None
        else core_sorted[:max_core_transition_lines]
    )
    for t in shown:
        lines.append(f"- {t.from_state} --{t.event}--> {t.to_state}")
    if (
        max_core_transition_lines is not None
        and len(core_sorted) > max_core_transition_lines
    ):
        lines.append("... truncated")

    positive = [t for t in spec.transitions if t.reward > 0]
    lines.append(f"transitions_with_reward>0: {len(positive)}")
    for t in positive:
        lines.append(
            f"- {t.from_state} --{t.event}--> {t.to_state} (reward={t.reward})"
        )

    return "\n".join(lines)
