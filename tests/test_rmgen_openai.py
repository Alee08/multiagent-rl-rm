import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from multiagent_rlrm.rmgen.pipeline import RMGenerationPipeline
from multiagent_rlrm.rmgen.providers import OpenAICompatLLMClient
from multiagent_rlrm.rmgen.validator import ValidationError


FIXTURES = Path(__file__).resolve().parent / "fixtures"


class DummyResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_openai_compat_success(monkeypatch):
    data = FIXTURES / "officeworld_simple.json"
    payload = {
        "choices": [
            {"message": {"content": data.read_text(encoding="utf-8")}},
        ]
    }
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *args, **kwargs: DummyResponse(payload)
    )
    client = OpenAICompatLLMClient(base_url="http://dummy", model="dummy", api_key=None)
    spec, rm = RMGenerationPipeline(client).run("task")
    assert spec.name == "office_simple"
    ns, rw = rm.get_reward_for_non_current_state("q0", "at(A)")
    assert ns == "q1" and rw == 0.0


def test_openai_compat_repair_path(monkeypatch):
    good = FIXTURES / "frozenlake_linear.json"
    responses = [
        DummyResponse({"choices": [{"message": {"content": "{not json}"}}]}),
        DummyResponse(
            {"choices": [{"message": {"content": good.read_text(encoding="utf-8")}}]}
        ),
    ]

    def fake_urlopen(*_args, **_kwargs):
        return responses.pop(0)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OpenAICompatLLMClient(base_url="http://dummy", model="dummy", api_key=None)
    spec, _rm = RMGenerationPipeline(client).run("repair-task")
    assert spec.name == "frozenlake_linear"
    assert len(responses) == 0  # both calls consumed


def test_openai_compat_rejects_nondeterminism(monkeypatch):
    bad = FIXTURES / "nondeterministic_rm.json"
    payload = {
        "choices": [
            {"message": {"content": bad.read_text(encoding="utf-8")}},
        ]
    }
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *_args, **_kwargs: DummyResponse(payload)
    )
    client = OpenAICompatLLMClient(base_url="http://dummy", model="dummy", api_key=None)
    with pytest.raises(ValidationError):
        RMGenerationPipeline(client).run("nondet")


def test_openai_compat_extracts_json_with_prefix(monkeypatch):
    data = FIXTURES / "warehouse_pickup_delivery.json"
    content = "Ecco il tuo RM:\n" + data.read_text(encoding="utf-8")
    payload = {"choices": [{"message": {"content": content}}]}
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *_args, **_kwargs: DummyResponse(payload)
    )
    client = OpenAICompatLLMClient(base_url="http://dummy", model="dummy", api_key=None)
    spec, _rm = RMGenerationPipeline(client).run("prefixed")
    assert spec.name == "warehouse_pickup_delivery"


def test_completion_added(monkeypatch):
    content = json.dumps(
        {
            "name": "partial",
            "env_id": "env",
            "version": "1.0",
            "states": ["q0", "q1"],
            "initial_state": "q0",
            "terminal_states": ["q1"],
            "event_vocabulary": ["e1", "e2"],
            "transitions": [
                {"from_state": "q0", "event": "e1", "to_state": "q1", "reward": 1}
            ],
        }
    )
    payload = {"choices": [{"message": {"content": content}}]}
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *_args, **_kwargs: DummyResponse(payload)
    )
    client = OpenAICompatLLMClient(base_url="http://dummy", model="dummy", api_key=None)
    spec, _rm = RMGenerationPipeline(client).run(
        "partial",
        complete=True,
        default_reward=0.5,
        terminal_reward_must_be_zero=False,
    )
    assert len(spec.transitions) == 4  # 2 states * 2 events
    # Check the explicit one remains reward 1 and to_state q1
    explicit = next(
        t
        for t in spec.transitions
        if t.from_state == "q0" and t.event == "e1" and t.to_state == "q1"
    )
    assert explicit.reward == 1.0
    # One of the completed ones
    completed = next(
        t for t in spec.transitions if t.from_state == "q1" and t.event == "e2"
    )
    assert completed.to_state == "q1" and completed.reward == 0.5
