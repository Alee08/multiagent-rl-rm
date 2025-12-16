import json
import subprocess
import sys
from pathlib import Path

import pytest

from multiagent_rlrm.rmgen.pipeline import RMGenerationPipeline
from multiagent_rlrm.rmgen.providers import MockLLMClient
from multiagent_rlrm.rmgen.validator import ValidationError


ROOT = Path(__file__).resolve().parent.parent


def test_pipeline_generates_and_exports(tmp_path):
    fixture = ROOT / "tests" / "fixtures" / "officeworld_simple.json"
    provider = MockLLMClient(fixture_path=fixture)
    pipeline = RMGenerationPipeline(provider)
    output = tmp_path / "rm_out.json"

    spec, rm = pipeline.run("officeworld-simple", output_path=output)

    assert output.exists()
    exported = json.loads(output.read_text())
    assert exported["name"] == "office_simple"
    assert spec.initial_state == "q0"
    # Check RM transition mapping is intact
    next_state, reward = rm.get_reward_for_non_current_state("q0", "at(A)")
    assert next_state == "q1"
    assert reward == 0.0
    next_state2, reward2 = rm.get_reward_for_non_current_state("q1", "at(G)")
    assert next_state2 == "q2"
    assert reward2 == 1.0


def test_rmgen_cli_mock_provider(tmp_path):
    fixture = ROOT / "tests" / "fixtures" / "frozenlake_linear.json"
    output = tmp_path / "cli_out.json"
    cmd = [
        sys.executable,
        "-m",
        "multiagent_rlrm.cli.rmgen",
        "--provider",
        "mock",
        "--mock-fixture",
        str(fixture),
        "--task",
        "linear run",
        "--output",
        str(output),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    data = json.loads(output.read_text())
    assert data["env_id"] == "frozenlake"
    assert data["states"] == ["q0", "q1", "q2"]


def test_pipeline_rejects_nondeterminism():
    fixture = ROOT / "tests" / "fixtures" / "nondeterministic_rm.json"
    provider = MockLLMClient(fixture_path=fixture)
    pipeline = RMGenerationPipeline(provider)
    with pytest.raises(ValidationError):
        pipeline.run("nondet")


def test_pipeline_rejects_invalid_schema():
    fixture = ROOT / "tests" / "fixtures" / "invalid_schema_rm.json"
    provider = MockLLMClient(fixture_path=fixture)
    pipeline = RMGenerationPipeline(provider)
    with pytest.raises(ValidationError):
        pipeline.run("invalid-schema")
