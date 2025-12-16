from __future__ import annotations

from typing import Dict, List, Set

from multiagent_rlrm.environments.office_world.config_office import config
from multiagent_rlrm.rmgen.normalize import normalize_event_key
from multiagent_rlrm.utils.utils import parse_office_world


def build_officeworld_context(map_name: str) -> Dict[str, object]:
    """
    Build an OfficeWorld "guardrail" context from the selected map layout.

    The context contains:
      - allowed_symbols: goal labels (e.g., A,B,...) plus object keywords (coffee, letter, ...)
      - allowed_events: list of allowed event tokens (both bare and at(...))
      - canonical_map: synonym -> canonical mapping (canonical form prefers at(...))
    """
    maps = config.get("maps", {})
    if map_name not in maps:
        raise ValueError(
            f"Unknown OfficeWorld map '{map_name}'. Available: {sorted(maps.keys())}"
        )

    layout = maps[map_name]["layout"]
    coordinates, goals, _walls = parse_office_world(layout)

    allowed_symbols: Set[str] = set(goals.keys())
    if "O" in goals:
        allowed_symbols.add("office")
    if coordinates.get("coffee"):
        allowed_symbols.add("coffee")
    if coordinates.get("letter"):
        allowed_symbols.update({"letter", "email"})

    allowed_events: Set[str] = set()
    canonical_map: Dict[str, str] = {}

    def add_alias(alias: str, canonical: str) -> None:
        allowed_events.add(alias)
        canonical_map[normalize_event_key(alias)] = canonical

    # Goal symbols from the map (A, B, C, ... including O).
    for sym in sorted(goals.keys()):
        canonical = f"at({sym})"
        add_alias(sym, canonical)
        add_alias(canonical, canonical)

    # Domain aliases: office -> O (when present).
    if "O" in goals:
        canonical_office = "at(O)"
        add_alias("office", canonical_office)
        add_alias("at(office)", canonical_office)

    # Objects (only if present in this map).
    if coordinates.get("coffee"):
        canonical_coffee = "at(coffee)"
        add_alias("coffee", canonical_coffee)
        add_alias("at(coffee)", canonical_coffee)

    if coordinates.get("letter"):
        canonical_letter = "at(letter)"
        add_alias("letter", canonical_letter)
        add_alias("email", canonical_letter)
        add_alias("at(letter)", canonical_letter)
        add_alias("at(email)", canonical_letter)

    return {
        "env_id": "officeworld",
        "map_name": map_name,
        "allowed_symbols": allowed_symbols,
        "allowed_events": sorted(allowed_events),
        "canonical_map": canonical_map,
    }
