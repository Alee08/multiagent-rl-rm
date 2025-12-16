import json

from multiagent_rlrm.environments.office_world.event_context import (
    build_officeworld_context,
)
from multiagent_rlrm.rmgen.io import load_rmspec
from multiagent_rlrm.rmgen.normalize import enforce_env_id
from multiagent_rlrm.rmgen.pipeline import RMGenerationPipeline
from multiagent_rlrm.rmgen.providers import MockLLMClient


def test_officeworld_env_id_is_enforced_in_pipeline_and_loader(tmp_path, capsys):
    # Provider outputs a typo env_id; OfficeWorld context must override it.
    raw = {
        "name": "typo_env",
        "env_id": "officework",
        "version": "1.0",
        "states": ["q0", "q1"],
        "initial_state": "q0",
        "terminal_states": ["q1"],
        "event_vocabulary": ["A"],
        "transitions": [
            {"from_state": "q0", "event": "A", "to_state": "q1", "reward": 1.0}
        ],
    }

    fixture = tmp_path / "typo.json"
    fixture.write_text(json.dumps(raw), encoding="utf-8")

    provider = MockLLMClient(fixture_path=fixture)
    pipeline = RMGenerationPipeline(provider)

    context = build_officeworld_context("map1")
    spec, _rm = pipeline.run(
        "task",
        normalize_context=context,
        enforce_env_id="officeworld",
        complete=True,
        max_positive_reward_transitions=1,
    )
    assert spec.env_id == "officeworld"

    # Loader path (mirrors OfficeWorld runner behavior)
    spec2 = load_rmspec(fixture)
    assert spec2.env_id == "officework"
    spec2 = enforce_env_id(spec2, "officeworld", reason="--rm-spec is set")
    assert spec2.env_id == "officeworld"

    out = capsys.readouterr().out
    assert "overriding env_id" in out


def test_officeworld_autofix_missing_state_added_and_validates(tmp_path):
    raw = {
        "name": "missing_state",
        "env_id": "officework",
        "version": "1.0",
        "states": ["q0", "q1"],
        "initial_state": "q0",
        "terminal_states": [],
        "event_vocabulary": ["A"],
        "transitions": [
            {"from_state": "q0", "event": "A", "to_state": "q4", "reward": 1.0}
        ],
    }

    fixture = tmp_path / "missing_state.json"
    fixture.write_text(json.dumps(raw), encoding="utf-8")

    provider = MockLLMClient(fixture_path=fixture)
    pipeline = RMGenerationPipeline(provider)
    context = build_officeworld_context("map1")

    spec, _rm = pipeline.run(
        "task",
        normalize_context=context,
        enforce_env_id="officeworld",
        complete=False,
        max_positive_reward_transitions=1,
    )
    assert "q4" in spec.states
    assert "q4" in spec.terminal_states


def test_officeworld_autofix_terminal_self_loop_reward_redirected(tmp_path):
    raw = {
        "name": "terminal_self_loop_reward",
        "env_id": "officework",
        "version": "1.0",
        "states": ["q0", "q1"],
        "initial_state": "q0",
        "terminal_states": ["q1"],
        "event_vocabulary": ["A"],
        "transitions": [
            {"from_state": "q0", "event": "A", "to_state": "q1", "reward": 0.0},
            {"from_state": "q1", "event": "A", "to_state": "q1", "reward": 1.0},
        ],
    }

    original_states = set(raw["states"])

    fixture = tmp_path / "terminal_self_loop_reward.json"
    fixture.write_text(json.dumps(raw), encoding="utf-8")

    provider = MockLLMClient(fixture_path=fixture)
    pipeline = RMGenerationPipeline(provider)
    context = build_officeworld_context("map1")

    spec, _rm = pipeline.run(
        "task",
        normalize_context=context,
        enforce_env_id="officeworld",
        complete=True,
        max_positive_reward_transitions=1,
        terminal_reward_must_be_zero=True,
    )

    added = set(spec.states) - original_states
    assert len(added) == 1
    new_terminal = next(iter(added))

    positive = [t for t in spec.transitions if t.reward > 0]
    assert len(positive) == 1
    assert positive[0].to_state == new_terminal
    assert positive[0].from_state not in set(spec.terminal_states)
    assert new_terminal in set(spec.terminal_states)
