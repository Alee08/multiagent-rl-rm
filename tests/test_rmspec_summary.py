from pathlib import Path

from multiagent_rlrm.rmgen.io import load_rmspec
from multiagent_rlrm.rmgen.summary import format_rmspec_summary
from multiagent_rlrm.rmgen.spec import RMSpec, TransitionSpec


def test_rmspec_summary_contains_env_id_and_positive_reward():
    fixture = Path(__file__).resolve().parent / "fixtures" / "officeworld_simple.json"
    spec = load_rmspec(fixture)
    summary = format_rmspec_summary(spec, agent_names=["agent0"], source=str(fixture))
    assert "env_id: officeworld" in summary
    assert "reward=1.0" in summary


def test_rmspec_summary_includes_core_transitions_excluding_self_loops():
    spec = RMSpec(
        name="core_only",
        env_id="officeworld",
        version="1.0",
        states=["q0", "q1"],
        initial_state="q0",
        terminal_states=["q1"],
        event_vocabulary=["at(C)", "at(E)"],
        transitions=[
            TransitionSpec(from_state="q0", event="at(C)", to_state="q1", reward=0.0),
            TransitionSpec(from_state="q0", event="at(E)", to_state="q0", reward=0.0),
        ],
    )

    summary = format_rmspec_summary(spec)
    assert "core_transitions_count: 1" in summary
    assert "q0 --at(C)--> q1" in summary
    assert "q0 --at(E)--> q0" not in summary
