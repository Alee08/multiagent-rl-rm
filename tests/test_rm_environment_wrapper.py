from multiagent_rlrm.multi_agent.reward_machine import RewardMachine
from multiagent_rlrm.multi_agent.wrappers.rm_environment_wrapper import (
    RMEnvironmentWrapper,
)


class DummyAction:
    def __init__(self, name):
        self.name = name


class DummyAlgo:
    use_qrm = False

    def choose_action(self, *_args, **_kwargs):
        return 0


class DummyEncoder:
    def encode(self, state, state_rm=None):
        pos = state["pos"]
        rm_state_idx = state_rm or 0
        return pos, {"s": pos, "q": rm_state_idx}


class DummyAgent:
    def __init__(self, rm):
        self.name = "agent"
        self._rm = rm
        self.encoder = DummyEncoder()
        self._algo = DummyAlgo()
        self._action = DummyAction("noop")

    def get_reward_machine(self):
        return self._rm

    def get_learning_algorithm(self):
        return self._algo

    def actions_idx(self, action):
        # Single action in this dummy setup
        return 0 if action.name == self._action.name else -1


class DummyEnv:
    def __init__(self, agent):
        self.agents = [agent]

    def reset(self, seed=None):
        del seed
        observations = {self.agents[0].name: {"pos": 0, "event": None}}
        infos = {self.agents[0].name: {"prev_s": observations[self.agents[0].name]}}
        return observations, infos

    def step(self, actions):
        del actions
        observations = {self.agents[0].name: {"pos": 1, "event": "goal"}}
        rewards = {self.agents[0].name: 0.5}
        terminations = {self.agents[0].name: False}
        truncations = {self.agents[0].name: False}
        infos = {self.agents[0].name: {"prev_s": {"pos": 0, "event": None}}}
        return observations, rewards, terminations, truncations, infos


class GoalEventDetector:
    def detect_event(self, state):
        return state.get("event")


def test_wrapper_adds_rm_reward_and_termination():
    transitions = {("q0", "goal"): ("qf", 1.0)}
    rm = RewardMachine(transitions, GoalEventDetector())
    agent = DummyAgent(rm)
    env = DummyEnv(agent)
    wrapper = RMEnvironmentWrapper(env, [agent])

    wrapper.reset(seed=123)
    observations, rewards, terminations, truncations, infos = wrapper.step(
        {agent.name: DummyAction("noop")}
    )

    assert observations[agent.name]["pos"] == 1
    assert rewards[agent.name] == 1.5  # env reward 0.5 + RM reward 1.0
    assert terminations[agent.name] is True  # RM reached final state
    assert truncations[agent.name] is False
    assert infos[agent.name]["prev_q"] == "q0"
    assert infos[agent.name]["q"] == "qf"


class AlgoWithQRM(DummyAlgo):
    use_qrm = True


def test_wrapper_generates_qrm_experiences_with_use_qrm_enabled():
    transitions = {("q0", "goal"): ("qf", 1.0), ("q0", None): ("q0", 0.0)}
    rm = RewardMachine(transitions, GoalEventDetector())
    agent = DummyAgent(rm)
    agent._algo = AlgoWithQRM()  # swap in algorithm with use_qrm=True
    env = DummyEnv(agent)
    wrapper = RMEnvironmentWrapper(env, [agent])

    wrapper.reset(seed=999)
    _, _, _, _, infos = wrapper.step({agent.name: DummyAction("noop")})

    qrm_exps = infos[agent.name]["qrm_experience"]
    assert isinstance(qrm_exps, list)
    assert len(qrm_exps) == len(rm.get_all_states()) - 1  # final state excluded


def test_wrapper_handles_multiple_agents_and_reward_modifier():
    transitions = {("q0", "goal"): ("qf", 1.0)}
    rm1 = RewardMachine(transitions, GoalEventDetector())
    rm2 = RewardMachine(transitions, GoalEventDetector())

    agent1 = DummyAgent(rm1)
    agent2 = DummyAgent(rm2)
    agent2.name = "agent2"
    env = DummyEnv(agent1)
    env.agents.append(agent2)

    def step_multi(actions):
        del actions
        observations = {
            agent1.name: {"pos": 1, "event": "goal"},
            agent2.name: {"pos": 2, "event": "goal"},
        }
        rewards = {agent1.name: 1.0, agent2.name: 2.0}
        terminations = {agent1.name: False, agent2.name: False}
        truncations = {agent1.name: False, agent2.name: False}
        infos = {
            agent1.name: {"prev_s": {"pos": 0, "event": None}},
            agent2.name: {"prev_s": {"pos": 0, "event": None}},
        }
        return observations, rewards, terminations, truncations, infos

    env.step = step_multi  # type: ignore

    wrapper = RMEnvironmentWrapper(env, [agent1, agent2])
    wrapper.reward_modifier = 2  # amplify RM reward

    wrapper.reset(seed=0)
    observations, rewards, terminations, truncations, infos = wrapper.step(
        {agent1.name: DummyAction("noop"), agent2.name: DummyAction("noop")}
    )

    assert rewards[agent1.name] == 1.0 + 2.0  # env + scaled RM reward
    assert rewards[agent2.name] == 2.0 + 2.0
    assert terminations[agent1.name] is True
    assert terminations[agent2.name] is True
    assert truncations[agent1.name] is False and truncations[agent2.name] is False
