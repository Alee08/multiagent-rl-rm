from multiagent_rlrm.environments.frozen_lake.event_context import (
    build_frozenlake_context,
)
from multiagent_rlrm.rmgen.normalize import normalize_rmspec_events


def test_frozenlake_normalize_events_accepts_bare_symbols():
    context = build_frozenlake_context("map1")

    spec = {
        "name": "bare_symbols_fl",
        "env_id": "frozenlake",
        "version": "1.0",
        "states": ["q0", "q1"],
        "initial_state": "q0",
        "terminal_states": ["q1"],
        "event_vocabulary": ["A", "B"],
        "transitions": [
            {"from_state": "q0", "event": "A", "to_state": "q1", "reward": 1.0},
        ],
    }

    normalized = normalize_rmspec_events(spec, context)
    assert normalized["event_vocabulary"] == ["at(A)", "at(B)"]
    assert normalized["transitions"][0]["event"] == "at(A)"
