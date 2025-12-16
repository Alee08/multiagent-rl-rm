import pytest

from multiagent_rlrm.rmgen.pipeline import RMGenerationPipeline
from multiagent_rlrm.rmgen.providers import MockLLMClient
from multiagent_rlrm.rmgen.validator import ValidationError


def make_fixture(content: str, tmp_path):
    path = tmp_path / "spec.json"
    path.write_text(content, encoding="utf-8")
    return path


def test_max_positive_reward_transitions_enforced(tmp_path):
    content = """
    {
      "name": "pos_limit",
      "env_id": "env",
      "version": "1.0",
      "states": ["q0"],
      "initial_state": "q0",
      "terminal_states": [],
      "event_vocabulary": ["e1", "e2"],
      "transitions": [
        {"from_state": "q0", "event": "e1", "to_state": "q0", "reward": 1},
        {"from_state": "q0", "event": "e2", "to_state": "q0", "reward": 1}
      ]
    }
    """
    fixture = make_fixture(content, tmp_path)
    provider = MockLLMClient(fixture_path=fixture)
    pipeline = RMGenerationPipeline(provider)
    with pytest.raises(ValidationError):
        pipeline.run(
            "task",
            max_positive_reward_transitions=1,
        )


def test_terminal_reward_must_be_zero(tmp_path):
    content = """
    {
      "name": "terminal_reward",
      "env_id": "env",
      "version": "1.0",
      "states": ["q0"],
      "initial_state": "q0",
      "terminal_states": ["q0"],
      "event_vocabulary": ["e1"],
      "transitions": [
        {"from_state": "q0", "event": "e1", "to_state": "q0", "reward": 1}
      ]
    }
    """
    fixture = make_fixture(content, tmp_path)
    provider = MockLLMClient(fixture_path=fixture)
    pipeline = RMGenerationPipeline(provider)
    with pytest.raises(ValidationError):
        pipeline.run(
            "task",
            terminal_reward_must_be_zero=True,
        )
