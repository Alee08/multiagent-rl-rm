from multiagent_rlrm.multi_agent.state_encoder import StateEncoder


class StateEncoderFrozenLake(StateEncoder):
    def encode(self, state, state_rm=None):
        """
        Codifies the current state, the Reward Machine state, and returns the necessary info.
        :param agent: The agent instance to access necessary agent-specific configurations.
        :param state: Dictionary representing the agent's state, including position.
        :return: A tuple (encoded_state, info) where info is a supplementary information dictionary.
        """
        num_rm_states = self.agent.get_reward_machine().numbers_state()
        pos_x, pos_y = state["pos_x"], state["pos_y"]
        rm_state_index = self.encode_rm_state(state_rm)
        max_x_value, max_y_value = (
            self.agent.ma_problem.grid_width,
            self.agent.ma_problem.grid_height,
        )

        pos_index = pos_y * max_x_value + pos_x
        encoded_state = pos_index * num_rm_states + rm_state_index

        total_states = max_x_value * max_y_value * num_rm_states
        if encoded_state >= total_states:
            raise ValueError("Encoded state index exceeds total state space size.")

        # Costruzione delle info
        info = {
            "s": pos_index,
            "q": rm_state_index,
        }
        # print(info, "weee")
        return encoded_state, info
