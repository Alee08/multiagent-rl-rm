from multiagent_rlrm.multi_agent.state_encoder import StateEncoder


class StateEncoderFrozenLake(StateEncoder):
    """Encode Frozen Lake agent positions (and RM states) into compact integer indices."""

    def encode(self, state, state_rm=None):
        """
        Encodes the current environment state and the Reward Machine state into a single index,
        and returns the encoded state together with auxiliary information.

        :param state: Dictionary representing the agent's state, including position.
                      Expected keys: "pos_x", "pos_y".
        :param state_rm: Current Reward Machine state (can be an index or a label depending on implementation).
        :return: A tuple (encoded_state, info) where:
                 - encoded_state is an integer representing the combined (position, RM state).
                 - info is a dictionary with additional information, e.g.:
                     {
                         "s": <position index on the grid>,
                         "q": <Reward Machine state index>,
                     }
        """
        num_rm_states = self.agent.get_reward_machine().numbers_state()
        pos_x, pos_y = state["pos_x"], state["pos_y"]
        rm_state_index = self.encode_rm_state(state_rm)
        max_x_value, max_y_value = (
            self.agent.ma_problem.grid_width,
            self.agent.ma_problem.grid_height,
        )

        # Encode the 2D grid position (pos_x, pos_y) into a single position index
        pos_index = pos_y * max_x_value + pos_x

        # Combine position index and Reward Machine state into a single global index
        encoded_state = pos_index * num_rm_states + rm_state_index

        # Compute the total number of possible states (grid cells × RM states)
        total_states = max_x_value * max_y_value * num_rm_states
        if encoded_state >= total_states:
            raise ValueError("Encoded state index exceeds total state space size.")

        # Build the auxiliary info dictionary
        info = {
            "s": pos_index,  # physical position index on the grid
            "q": rm_state_index,  # Reward Machine state index
        }
        # print(info, "weee")
        return encoded_state, info

    def decode(self, encoded_state):
        """
        Decodes the integer state index `encoded_state` into its components:
          - pos_x, pos_y (physical coordinates on the grid)
          - Reward Machine state (as index or as string label).

        Returns a tuple (state_dict, info_dict) where:
          - state_dict = {"pos_x": ..., "pos_y": ...}
          - info_dict = {"q": <Reward Machine state name or index>}
        """
        num_rm_states = self.agent.get_reward_machine().numbers_state()
        max_x_value = self.agent.ma_problem.grid_width
        max_y_value = self.agent.ma_problem.grid_height

        # 1) Recover the Reward Machine state index and the physical position index (pos_index)
        rm_state_index = encoded_state % num_rm_states
        pos_index = encoded_state // num_rm_states

        # 2) Recover pos_y and pos_x from pos_index
        pos_y = pos_index // max_x_value
        pos_x = pos_index % max_x_value

        # 3) Build the physical state dictionary
        state_dict = {
            "pos_x": pos_x,
            "pos_y": pos_y,
        }

        # 4) Retrieve the Reward Machine state name (string) from its index
        rm_state_str = self.agent.get_reward_machine().get_state_name_from_idx(
            rm_state_index
        )

        # 5) Build the info dictionary with the Reward Machine state
        info_dict = {"q": rm_state_str}  # Reward Machine state

        return state_dict, info_dict
