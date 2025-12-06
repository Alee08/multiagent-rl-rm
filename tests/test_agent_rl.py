from multiagent_rlrm.multi_agent.agent_rl import AgentRL
from multiagent_rlrm.multi_agent.reward_machine import RewardMachine
from multiagent_rlrm.multi_agent.state_encoder import StateEncoder
from multiagent_rlrm.multi_agent.action_encoder import ActionEncoder
from multiagent_rlrm.multi_agent.action_rl import ActionRL


class DummyEventDetector:
    def detect_event(self, _state):
        return None


class DummyProblem:
    grid_width = 2
    grid_height = 2


class DummyStateEncoder(StateEncoder):
    def encode(self, state, state_rm=None):
        rm_state = state_rm or self.agent.get_reward_machine().get_current_state()
        rm_idx = self.agent.get_reward_machine().get_state_index(rm_state)
        encoded_state = state["encoded"]
        return encoded_state, {"s": encoded_state, "q": rm_idx}


class DummyActionEncoder(ActionEncoder):
    def build_actions(self):
        self.agent.add_action(ActionRL("a0"))
        self.agent.add_action(ActionRL("a1"))


class DummyAlgo:
    use_qrm = False

    def __init__(self):
        self.calls = []

    def choose_action(self, encoded_state, best=False, rng=None, **kwargs):
        return 1  # pick second action for determinism

    def update(
        self, encoded_state, encoded_next_state, action, reward, terminated, **kwargs
    ):
        self.calls.append(
            {
                "s": encoded_state,
                "sn": encoded_next_state,
                "a": action,
                "r": reward,
                "terminated": terminated,
                "info": kwargs.get("info"),
            }
        )
        return "updated"


def test_agent_select_action_and_update_policy():
    transitions = {("q0", None): ("q0", 0)}
    rm = RewardMachine(transitions, DummyEventDetector())
    # Bypass AgentRL.__init__ to avoid needing a full Unified Planning environment
    agent = AgentRL.__new__(AgentRL)
    object.__setattr__(agent, "_name", "ag")  # bypass immutability guard
    agent.ma_problem = DummyProblem()
    agent.reward_machine = rm
    agent.actions_dict = {}
    agent.learning_algorithm = None
    agent.message_conditions = None
    agent.messages = {}
    agent.message_sent = False
    agent.position = None
    agent.state = {}
    agent.actions_ = []
    agent.initial_position = {}
    agent.initial_state = {}
    agent.rm_state = None
    agent.next_rm_state = None
    agent.encoder = None
    agent.action_encoder = None
    agent.add_action_encoder(DummyActionEncoder(agent))
    agent.add_state_encoder(DummyStateEncoder(agent))
    algo = DummyAlgo()
    agent.set_learning_algorithm(algo)

    action = agent.select_action({"encoded": 0})
    assert action.name == "a1"  # index 1 returned by DummyAlgo.choose_action

    term = agent.update_policy(
        state={"encoded": 0},
        action=action,
        reward=0.5,
        next_state={"encoded": 1},
        terminated=True,
        infos={"prev_q": "q0", "q": "q0", "reward_machine": rm},
    )

    assert term == "updated"
    assert algo.calls[0]["a"] == 1
    assert algo.calls[0]["s"] == 0
    assert algo.calls[0]["sn"] == 1
    assert algo.calls[0]["terminated"] is True
    assert algo.calls[0]["info"]["q"] == 0  # encoded RM state index
