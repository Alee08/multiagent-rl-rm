from __future__ import annotations

from typing import Dict, Set

from multiagent_rlrm.environments.frozen_lake.config_frozen_lake import config
from multiagent_rlrm.rmgen.normalize import normalize_event_key
from multiagent_rlrm.utils.utils import parse_map_emoji


def build_frozenlake_context(map_name: str) -> Dict[str, object]:
    """
    Build a FrozenLake "guardrail" context from the selected map layout.

    The context contains:
      - allowed_symbols: goal labels (e.g., A,B,1,2,...) derived from the map
      - allowed_events: list of allowed event tokens (both bare and at(...))
      - canonical_map: synonym -> canonical mapping (canonical form prefers at(...))

    Note: holes are intentionally excluded from the RM event vocabulary.
    """
    maps = config.get("maps", {})
    if map_name not in maps:
        raise ValueError(
            f"Unknown FrozenLake map '{map_name}'. Available: {sorted(maps.keys())}"
        )

    layout = maps[map_name]["layout"]
    _holes, goals, _dims = parse_map_emoji(layout)

    allowed_symbols: Set[str] = set(goals.keys())
    allowed_events: Set[str] = set()
    canonical_map: Dict[str, str] = {}

    def add_alias(alias: str, canonical: str) -> None:
        allowed_events.add(alias)
        canonical_map[normalize_event_key(alias)] = canonical

    for sym in sorted(allowed_symbols):
        canonical = f"at({sym})"
        add_alias(sym, canonical)
        add_alias(canonical, canonical)

    return {
        "env_id": "frozenlake",
        "map_name": map_name,
        "allowed_symbols": allowed_symbols,
        "allowed_events": sorted(allowed_events),
        "canonical_map": canonical_map,
    }
