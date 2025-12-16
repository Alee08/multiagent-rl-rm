import inspect

from multiagent_rlrm.multi_agent.reward_machine import RewardMachine


def test_reward_machine_init_signature_is_stable():
    """
    Guardrail: the RewardMachine constructor signature is relied upon by rmgen.
    Fail fast if it changes.
    """
    sig = inspect.signature(RewardMachine.__init__)
    params = list(sig.parameters.values())
    names = [p.name for p in params]
    assert names[:3] == ["self", "transitions", "event_detector"]
    # transitions and event_detector must remain required positional args
    assert params[1].default is inspect._empty
    assert params[2].default is inspect._empty
