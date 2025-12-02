# multiagent_rlrm/environments/frozen_lake/action_encoder_frozen_lake.py

from multiagent_rlrm.multi_agent.action_encoder import ActionEncoder
from multiagent_rlrm.multi_agent.action_rl import ActionRL


class ActionEncoderFrozenLake(ActionEncoder):
    """
    Action encoder for the Frozen Lake domain.
    Registers only symbolic actions; the dynamics are handled by the environment.
    """

    def build_actions(self):
        self.agent.add_action(ActionRL("up"))
        self.agent.add_action(ActionRL("down"))
        self.agent.add_action(ActionRL("left"))
        self.agent.add_action(ActionRL("right"))
