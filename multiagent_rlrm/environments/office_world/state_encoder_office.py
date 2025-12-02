from multiagent_rlrm.multi_agent.state_encoder import StateEncoder


class StateEncoderOfficeWorld(StateEncoder):
    """Encodes Office World agent states into integer indices for tabular algorithms."""

    def encode(self, state, state_rm=None):
        """
        Encodes grid position and reward-machine state into a single integer.

        :param state: Dictionary with keys like "pos_x" and "pos_y".
        :param state_rm: Optional reward-machine state identifier.
        :return: Tuple (encoded_state, info_dict) where info_dict includes spatial and RM indices.
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

        # Build auxiliary info
        info = {
            "s": pos_index,
            "q": rm_state_index,
        }
        return encoded_state, info

    def decode(self, encoded_state):
        """
        Decodes an integer state into grid coordinates and reward-machine state.

        :param encoded_state: Combined state index.
        :return: Tuple (decoded_state_dict, rm_state_label).
        """
        rm = self.agent.get_reward_machine()
        num_rm_states = rm.numbers_state()

        max_x_value = self.agent.ma_problem.grid_width
        max_y_value = self.agent.ma_problem.grid_height
        total_states = max_x_value * max_y_value * num_rm_states

        if encoded_state < 0 or encoded_state >= total_states:
            raise ValueError(
                f"Encoded state {encoded_state} out of range [0..{total_states - 1}]."
            )

        rm_state_index = encoded_state % num_rm_states
        pos_index = encoded_state // num_rm_states

        pos_x = pos_index % max_x_value
        pos_y = pos_index // max_x_value

        rm_state = rm.get_state_from_index(rm_state_index)

        decoded_state = {"pos_x": pos_x, "pos_y": pos_y}

        return decoded_state, rm_state
