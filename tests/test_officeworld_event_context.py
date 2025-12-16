import json

from multiagent_rlrm.environments.office_world.event_context import (
    build_officeworld_context,
)
from multiagent_rlrm.environments.office_world.config_office import (
    config as office_config,
)
from multiagent_rlrm.environments.frozen_lake.detect_event import PositionEventDetector
from multiagent_rlrm.rmgen.io import compile_reward_machine, load_rmspec
from multiagent_rlrm.rmgen.normalize import normalize_rmspec_events
from multiagent_rlrm.utils.utils import parse_office_world


def test_normalize_events_accepts_bare_symbols():
    context = build_officeworld_context("map1")

    spec = {
        "name": "bare_symbols",
        "env_id": "officeworld",
        "version": "1.0",
        "states": ["q0", "q1", "q2", "q3", "q4"],
        "initial_state": "q0",
        "terminal_states": ["q4"],
        "event_vocabulary": ["A", "B", "C", "D"],
        "transitions": [
            {"from_state": "q0", "event": "A", "to_state": "q1", "reward": 0},
            {"from_state": "q1", "event": "B", "to_state": "q2", "reward": 0},
            {"from_state": "q2", "event": "C", "to_state": "q3", "reward": 0},
            {"from_state": "q3", "event": "D", "to_state": "q4", "reward": 1},
        ],
    }

    normalized = normalize_rmspec_events(spec, context)
    assert normalized["event_vocabulary"] == ["at(A)", "at(B)", "at(C)", "at(D)"]
    assert [t["event"] for t in normalized["transitions"]] == [
        "at(A)",
        "at(B)",
        "at(C)",
        "at(D)",
    ]


def test_normalize_events_office_synonym():
    context = build_officeworld_context("map1")

    spec = {
        "name": "office_synonym",
        "env_id": "officeworld",
        "version": "1.0",
        "states": ["q0", "q1"],
        "initial_state": "q0",
        "terminal_states": ["q1"],
        "event_vocabulary": ["at(office)"],
        "transitions": [
            {"from_state": "q0", "event": "at(office)", "to_state": "q1", "reward": 1},
        ],
    }

    normalized = normalize_rmspec_events(spec, context)
    assert normalized["event_vocabulary"] == ["at(O)"]
    assert normalized["transitions"][0]["event"] == "at(O)"


def test_office_main_rm_spec_normalization_smoke(tmp_path):
    # Minimal RM spec using a bare symbol event; OfficeWorld runner should accept it
    # after normalization against the selected map.
    spec_path = tmp_path / "rm.json"
    spec_path.write_text(
        json.dumps(
            {
                "name": "smoke",
                "env_id": "officeworld",
                "version": "1.0",
                "states": ["q0", "q1"],
                "initial_state": "q0",
                "terminal_states": ["q1"],
                "event_vocabulary": ["A"],
                "transitions": [
                    {
                        "from_state": "q0",
                        "event": "A",
                        "to_state": "q1",
                        "reward": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    spec = load_rmspec(spec_path)
    context = build_officeworld_context("map1")
    spec = normalize_rmspec_events(spec, context)

    layout = office_config["maps"]["map1"]["layout"]
    coordinates, goals, _walls = parse_office_world(layout)

    event_mapping = {}
    for label, pos in goals.items():
        event_mapping[label] = pos
        event_mapping[f"at({label})"] = pos
    if "O" in goals:
        event_mapping["office"] = goals["O"]
        event_mapping["at(office)"] = goals["O"]

    coffee_positions = coordinates.get("coffee") or []
    if coffee_positions:
        event_mapping["coffee"] = list(coffee_positions)
        event_mapping["at(coffee)"] = list(coffee_positions)

    letter_positions = coordinates.get("letter") or []
    if letter_positions:
        event_mapping["letter"] = list(letter_positions)
        event_mapping["email"] = list(letter_positions)
        event_mapping["at(letter)"] = list(letter_positions)
        event_mapping["at(email)"] = list(letter_positions)

    event_positions = set()
    for mapped in event_mapping.values():
        if isinstance(mapped, (list, set, frozenset)):
            event_positions.update(mapped)
        else:
            event_positions.add(mapped)

    event_detector = PositionEventDetector(event_positions)
    rm = compile_reward_machine(
        spec,
        event_detector=event_detector,
        event_mapping=event_mapping,
    )

    next_state, reward = rm.get_reward_for_non_current_state("q0", goals["A"])
    assert next_state == "q1"
    assert reward == 1.0
