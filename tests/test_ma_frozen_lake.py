import numpy as np

from multiagent_rlrm.environments.frozen_lake.action_encoder_frozen_lake import (
    ActionEncoderFrozenLake,
)
from multiagent_rlrm.environments.frozen_lake.detect_event import PositionEventDetector
from multiagent_rlrm.environments.frozen_lake.ma_frozen_lake import MultiAgentFrozenLake
from multiagent_rlrm.environments.frozen_lake.state_encoder_frozen_lake import (
    StateEncoderFrozenLake,
)
from multiagent_rlrm.multi_agent.agent_rl import AgentRL
from multiagent_rlrm.multi_agent.reward_machine import RewardMachine
from multiagent_rlrm.multi_agent.wrappers.rm_environment_wrapper import (
    RMEnvironmentWrapper,
)


class StubAgent:
    def __init__(self, name, x=0, y=0):
        self.name = name
        self._pos = (x, y)
        self.initial_position = (x, y)
        self.state = {"pos_x": x, "pos_y": y}

    def get_position(self):
        return self._pos

    def set_initial_position(self, x, y):
        self.initial_position = (x, y)
        self.set_position(x, y)

    def set_position(self, x, y):
        self._pos = (x, y)
        self.state = {"pos_x": x, "pos_y": y}

    def get_state(self):
        return {"pos_x": self._pos[0], "pos_y": self._pos[1]}

    def get_learning_algorithm(self):
        return None

    def reset(self):
        self.set_position(*self.initial_position)


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


def test_reset_infos_are_keyed_by_agent_name():
    env = MultiAgentFrozenLake(width=2, height=2, holes=[])
    agent = StubAgent("a", x=0, y=0)
    env.agents.append(agent)

    observations, infos = env.reset(seed=123)

    assert set(observations) == {"a"}
    assert set(infos) == {"a"}


def test_wrapper_get_mdp_supports_frozen_lake():
    env = MultiAgentFrozenLake(width=2, height=2, holes=[])
    agent = AgentRL("a", env)
    agent.set_initial_position(0, 0)
    agent.add_state_encoder(StateEncoderFrozenLake(agent))
    agent.add_action_encoder(ActionEncoderFrozenLake(agent))
    rm = RewardMachine({("q0", (1, 0)): ("qf", 1)}, PositionEventDetector({(1, 0)}))
    agent.set_reward_machine(rm)
    env.add_agent(agent)

    all_p, all_num_states, all_num_actions = RMEnvironmentWrapper(
        env, [agent]
    ).get_mdp(seed=123)

    assert all_num_states["a"] == 8
    assert all_num_actions["a"] == 4
    assert set(all_p) == {"a"}
