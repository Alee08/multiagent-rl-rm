import numpy as np

from multiagent_rlrm.environments.frozen_lake.ma_frozen_lake import MultiAgentFrozenLake


class StubAgent:
    def __init__(self, name, x=0, y=0):
        self.name = name
        self._pos = (x, y)
        self.state = {"pos_x": x, "pos_y": y}

    def get_position(self):
        return self._pos

    def set_position(self, x, y):
        self._pos = (x, y)
        self.state = {"pos_x": x, "pos_y": y}

    def get_state(self):
        return {"pos_x": self._pos[0], "pos_y": self._pos[1]}


def test_apply_action_respects_boundaries():
    env = MultiAgentFrozenLake(width=2, height=2, holes=[])
    agent = StubAgent("a", x=0, y=0)

    env.apply_action(agent, "left")
    assert agent.get_position() == (0, 0)
    env.apply_action(agent, "up")
    assert agent.get_position() == (0, 0)

    env.apply_action(agent, "right")
    assert agent.get_position() == (1, 0)
    env.apply_action(agent, "down")
    assert agent.get_position() == (1, 1)


def test_get_stochastic_action_respects_action_map():
    env = MultiAgentFrozenLake(width=3, height=3, holes=[])
    agent = StubAgent("a")

    np.random.seed(0)
    chosen = env.get_stochastic_action(agent, "left")
    assert chosen in {"left", "up", "down"}

    env.delay_action = True
    np.random.seed(1)
    chosen_delay = env.get_stochastic_action(agent, "up")
    assert chosen_delay in {"wait", "up", "left", "right"}
