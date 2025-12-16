import json

from multiagent_rlrm.cli import rmgen as rmgen_cli
from multiagent_rlrm.environments.office_world.event_context import (
    build_officeworld_context,
)


class DummyResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_rmgen_context_injects_allowed_events(monkeypatch, tmp_path):
    captured = {}

    # Minimal valid spec; events will be normalized against the OfficeWorld map context.
    content = json.dumps(
        {
            "name": "office_ctx",
            "env_id": "officeworld",
            "version": "1.0",
            "states": ["q0", "q1"],
            "initial_state": "q0",
            "terminal_states": ["q1"],
            "event_vocabulary": ["A"],
            "transitions": [
                {"from_state": "q0", "event": "A", "to_state": "q1", "reward": 1}
            ],
        }
    )

    def fake_urlopen(req, *_args, **_kwargs):
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return DummyResponse({"choices": [{"message": {"content": content}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    output = tmp_path / "rm_out.json"
    rc = rmgen_cli.main(
        [
            "--provider",
            "openai_compat",
            "--base-url",
            "http://dummy",
            "--model",
            "dummy",
            "--context",
            "officeworld",
            "--map",
            "map1",
            "--task",
            "A -> C -> B -> D, reward on D",
            "--output",
            str(output),
        ]
    )
    assert rc == 0

    messages = captured["payload"]["messages"]
    user_msg = next(m["content"] for m in messages if m["role"] == "user")
    assert "Allowed events" in user_msg

    context = build_officeworld_context("map1")
    # A couple of representative tokens derived from the map.
    assert "- at(A)" in user_msg
    assert "- coffee" in user_msg
    assert "- at(coffee)" in user_msg
    assert "- at(O)" in user_msg


def test_rmgen_officeworld_safe_defaults_applied(monkeypatch, tmp_path):
    captured = {"pipeline_kwargs": {}, "payload": None}

    content = json.dumps(
        {
            "name": "partial_office",
            "env_id": "officeworld",
            "version": "1.0",
            "states": ["q0", "q1"],
            "initial_state": "q0",
            "terminal_states": ["q1"],
            "event_vocabulary": ["A", "B"],
            "transitions": [
                {"from_state": "q0", "event": "A", "to_state": "q1", "reward": 1}
            ],
        }
    )

    def fake_urlopen(req, *_args, **_kwargs):
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return DummyResponse({"choices": [{"message": {"content": content}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    from multiagent_rlrm.rmgen.pipeline import RMGenerationPipeline

    original_run = RMGenerationPipeline.run

    def run_wrapper(self, task, output_path=None, **kwargs):
        captured["pipeline_kwargs"] = dict(kwargs)
        return original_run(self, task, output_path=output_path, **kwargs)

    monkeypatch.setattr(RMGenerationPipeline, "run", run_wrapper)

    output = tmp_path / "rm_out.json"
    rc = rmgen_cli.main(
        [
            "--provider",
            "openai_compat",
            "--base-url",
            "http://dummy",
            "--model",
            "dummy",
            "--context",
            "officeworld",
            "--map",
            "map1",
            "--task",
            "A then B, reward on B",
            "--output",
            str(output),
        ]
    )
    assert rc == 0
    assert captured["payload"]["temperature"] == 0.0
    assert captured["pipeline_kwargs"]["complete"] is True
    assert captured["pipeline_kwargs"]["max_positive_reward_transitions"] == 1

    data = json.loads(output.read_text(encoding="utf-8"))
    assert len(data["transitions"]) == len(data["states"]) * len(
        data["event_vocabulary"]
    )
    keys = {(t["from_state"], t["event"]) for t in data["transitions"]}
    assert len(keys) == len(data["transitions"])
