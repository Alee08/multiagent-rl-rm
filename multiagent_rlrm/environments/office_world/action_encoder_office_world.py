from multiagent_rlrm.multi_agent.action_encoder import ActionEncoder
from multiagent_rlrm.multi_agent.action_rl import ActionRL


class ActionEncoderOfficeWorld(ActionEncoder):
    """Registers symbolic movement actions for Office World agents."""

    def build_actions(self):
        self.agent.add_action(ActionRL("up"))
        self.agent.add_action(ActionRL("down"))
        self.agent.add_action(ActionRL("left"))
        self.agent.add_action(ActionRL("right"))
