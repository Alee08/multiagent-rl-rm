from pathlib import Path

from multiagent_rlrm.multi_agent.reward_machine import RewardMachine
from multiagent_rlrm.rmgen.io import compile_reward_machine, load_rmspec


ROOT = Path(__file__).resolve().parent.parent


def test_load_and_compile_rmspec_fixture():
    spec_path = ROOT / "tests" / "fixtures" / "officeworld_simple.json"
    spec = load_rmspec(spec_path)
    rm = compile_reward_machine(spec)

    assert isinstance(rm, RewardMachine)
    next_state, reward = rm.get_reward_for_non_current_state("q1", "at(G)")
    assert next_state == "q2"
    assert reward == 1.0
