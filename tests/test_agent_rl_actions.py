from multiagent_rlrm.multi_agent.agent_rl import AgentRL
from multiagent_rlrm.multi_agent.action_rl import ActionRL
from unified_planning.exceptions import UPValueError


def test_actions_idx_handles_multiple_actions():
    agent = AgentRL.__new__(AgentRL)
    object.__setattr__(agent, "_name", "ag")
    agent.actions_ = [ActionRL("a0"), ActionRL("a1")]
    agent.actions_dict = {}

    # Should rebuild mapping each call
    assert agent.actions_idx(agent.actions_[0]) == 0
    assert agent.actions_idx(agent.actions_[1]) == 1


def test_action_raises_up_value_error_for_unknown_action():
    agent = AgentRL.__new__(AgentRL)
    object.__setattr__(agent, "_name", "ag")
    agent.actions_ = [ActionRL("a0")]

    try:
        agent.action("missing")
    except UPValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("Expected UPValueError")
