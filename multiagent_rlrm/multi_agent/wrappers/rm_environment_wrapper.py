from multiagent_rlrm.learning_algorithms.qlearning import QLearning


class RMEnvironmentWrapper:
    """
    A wrapper for an environment to manage interactions with Reward Machines (RM) for multiple agents.

    This wrapper intercepts the environment's step and reset functions to update the Reward Machines
    and their associated states and rewards for each agent.
    """

    def __init__(self, env, agents):
        """
        Initialize the RMEnvironmentWrapper.

        Args:
            env (BaseEnvironment): The environment to wrap.
            agents (list): A list of agents interacting with the environment.
        """
        self.env = env
        self.agents = agents  # List of agents
        self.reward_modifier = 1  # reward modifier to keep it close to 1 depending on length of optimal traces

    def reset(self, seed):
        """
        Reset the environment and the Reward Machines to their initial states.

        Returns:
            tuple: A tuple containing observations and infos dictionaries.
        """
        observations, infos = self.env.reset(seed)
        for agent in self.agents:
            agent.get_reward_machine().reset_to_initial_state()
        return observations, infos

    def step(self, actions):
        """
        Perform a step in the environment and update the Reward Machines.

        Args:
            actions (dict): A dictionary mapping agent names to their respective actions.

        Returns:
            tuple: A tuple containing observations, rewards, terminations, truncations, and infos dictionaries.
        """
        observations, rewards, env_terminations, env_truncations, infos = self.env.step(
            actions
        )
        for agent in self.agents:
            rm = agent.get_reward_machine()
            # current_state = infos[agent.name]["prev_s"]
            current_state = infos[agent.name].get("prev_s", observations[agent.name])
            # next_state_ = infos[agent.name]["s"]  # observations[agent.name] == next_state
            next_state_ = infos[agent.name].get("s", observations[agent.name])
            next_state = observations[agent.name]
            state_rm = rm.get_current_state()
            reward_rm = rm.step(next_state)  # Get the reward from the RM
            new_state_rm = rm.get_current_state()

            # TODO CHECK - keeps final discounted reward close to 1
            reward_rm *= self.reward_modifier
            # print(f" -- debug {self.reward_modifier} --")

            # print(reward_rm, next_state, new_state_rm, "oooooooo")

            infos[agent.name]["RQ"] = reward_rm  # Add RM reward information
            infos[agent.name]["prev_q"] = state_rm  # RM state before the event
            infos[agent.name]["q"] = new_state_rm  # RM state after the event

            infos[agent.name]["reward_machine"] = rm

            use_qrm = False
            try:
                use_qrm = agent.get_learning_algorithm().use_qrm
            except:
                pass

            if use_qrm:
                qrm_experiences = self._get_qrm_experiences(
                    agent,
                    current_state,
                    next_state,
                    actions[agent.name],
                    rewards[agent.name],  # env reward
                    new_state_rm,
                    env_terminations[agent.name],
                )
                infos[agent.name]["qrm_experience"] = qrm_experiences

            rewards[agent.name] += reward_rm  # total reward

        rm_terminations = self.check_terminations()

        terminations = {}
        for agent in self.agents:
            terminations[agent.name] = (
                env_terminations[agent.name] or rm_terminations[agent.name]
            )
            infos[agent.name]["env_terminated"] = env_terminations[agent.name]
            infos[agent.name]["rm_terminated"] = rm_terminations[agent.name]
        truncations = env_truncations

        return observations, rewards, terminations, truncations, infos

    def check_terminations(self):
        """
        Check for terminations and truncations in the environment and the Reward Machines.

        Returns:
            tuple: A tuple containing terminations and truncations dictionaries.
        """
        rm_terminations = {}
        for agent in self.agents:
            rm = agent.get_reward_machine()
            rm_terminations[agent.name] = rm.get_current_state() == rm.get_final_state()
            # truncations[agent.name] = True    # FIX error

        return rm_terminations

    def _get_qrm_experiences(
        self,
        agent,
        current_state,
        next_state,
        action,
        env_reward,
        next_rm_state,
        env_termination,
    ):
        qrm_experiences = []
        action_index = agent.actions_idx(action)
        rm = agent.get_reward_machine()
        all_states = rm.get_all_states()[:-1]
        final_state = rm.get_final_state()
        for state_rm in all_states:
            event = rm.event_detector.detect_event(next_state)
            (
                hypothetical_next_state,
                hypothetical_reward,
            ) = rm.get_reward_for_non_current_state(state_rm, event)

            # Aggiungi ulteriori debug per vedere quale evento viene rilevato
            # print(f"State RM: {state_rm}, Event: {event}, Hypothetical Next State: {hypothetical_next_state}, Hypothetical Reward: {hypothetical_reward}")

            # Salta se lo stato ipotetico successivo è None
            if hypothetical_next_state is None:
                hypothetical_next_state = state_rm
            # print(current_state, state_rm, next_state, hypothetical_next_state, "weeeeeeeeeeeeee")
            encoded_state, enc_state_info = agent.encoder.encode(
                current_state, state_rm
            )
            encoded_next_state, enc_next_state_info = agent.encoder.encode(
                next_state, hypothetical_next_state
            )
            # print(f"QRM Experience: state_rm={state_rm}, event={event}, hypo_next_state={hypothetical_next_state}, hypo_reward={hypothetical_reward}")
            """if event != None:
                print("current_state:", current_state, "next_state:", next_state, "state_rm:", state_rm, "hypothetical_next_state:", hypothetical_next_state, "hypothetical_reward:", hypothetical_reward)"""

            # Verifica se lo stato ipotetico successivo è uno stato terminale
            # done = hypothetical_next_state == agent.get_reward_machine().get_final_state()
            rm_done = hypothetical_next_state == final_state
            # done = rm_done or env_done

            """if agent.ma_problem.agent_fail[agent.name] or agent.ma_problem.timestep > 1000:
                done = True"""

            qrm_experience = (
                encoded_state,
                action_index,
                env_reward + hypothetical_reward,  # total reward for the encoded state
                encoded_next_state,
                env_termination or rm_done,  # TODO check ...
                enc_state_info["s"],
                enc_state_info["q"],
                enc_next_state_info["s"],
                enc_next_state_info["q"],
                hypothetical_reward,  # only rm reward
            )
            qrm_experiences.append(qrm_experience)
            """if rm_done:
                print("qrm_experience:", qrm_experience, "env_done:", env_done, "rm_done", rm_done)
                breakpoint()"""
            # print(f"QRM Experience: state_rm={state_rm}, event={event}, hypo_next_state={hypothetical_next_state}, hypo_reward={hypothetical_reward}, encoded_state={encoded_state}, encoded_next_state={encoded_next_state}, done={done}")
        return qrm_experiences

    def get_mdp(self, seed):
        """
        Extracts the Markov Decision Process (MDP) for each agent in a multi-agent environment.

        The function builds the transition model (`P`), the number of states (`num_states`),
        and the number of actions (`num_actions`) for each agent, considering their Reward Machine (RM) states
        and environment dynamics.

        Parameters:
        - seed: A random seed used to ensure consistent behavior during environment resets.

        Returns:
        - all_P: A dictionary where the keys are agent names and the values are the transition models `P`.
                 `P` is a dictionary mapping state IDs to a dictionary of actions and their transitions.
        - all_num_states: A dictionary with the total number of states for each agent.
        - all_num_actions: A dictionary with the total number of actions for each agent.

        Process:
        1. **Initialization**:
            - Retrieve the environment's dimensions (`width`, `height`) and list of agents.
            - For each agent, determine the number of states and actions based on the environment grid and RM states.
            - Initialize the transition model `P` as a nested dictionary.

        2. **Iterate over States**:
            - Decode the state ID into grid position and RM state for the agent.
            - Check if the state is terminal. If terminal:
              - Create a self-loop with the terminal reward.

        3. **Generate Transitions for Actions**:
            - Reset the environment and set the agent's state.
            - For each action:
              - Retrieve possible sub-actions and their probabilities.
              - Simulate the environment by stepping with the agent's action and random actions for others.
              - Record the resulting state, reward, and terminal status in `P`.

        4. **Store Results**:
            - Save the transition model `P`, the number of states, and the number of actions for each agent.

        5. **Restore Initial State**:
            - Reset the environment to its initial state for consistency.

        Notes:
        - This function accommodates stochastic environments where actions may have multiple outcomes.
        - Errors during environment steps are logged for debugging purposes but do not interrupt the process.
        - The final MDP includes the dynamics of both the agent's RM states and the physical grid.
        """
        self.env.stochastic = False
        width = self.env.map_width
        height = self.env.map_height
        agents = self.env.agents
        # Each agent has its own Reward Machine and its own MDP
        all_P = {}
        all_num_states = {}
        all_num_actions = {}
        for agent in agents:
            rm = agent.get_reward_machine()
            num_rm_states = rm.numbers_state()
            num_states = width * height * num_rm_states
            actions_list = agent.get_actions()
            num_actions = len(actions_list)

            # P = [[[] for _ in range(num_actions)] for _ in range(num_states)]
            P = {
                s_id: {a_idx: [] for a_idx in range(num_actions)}
                for s_id in range(num_states)
            }

            print(
                f"MDP construction for agent {agent.name}: Total states={num_states}, Actions={num_actions}"
            )
            for cod_state in range(num_states):
                decoded_state, rm_state = agent.encoder.decode(cod_state)
                # s_id = self.env.state_to_id(state)
                pos_x, pos_y = decoded_state["pos_x"], decoded_state["pos_y"]

                is_terminal, terminal_reward = self.env.is_terminal_state_mdp(
                    agent, pos_x, pos_y, rm_state
                )
                if is_terminal:
                    # terminal state -> autoloop
                    for a_idx in range(num_actions):
                        P[cod_state][a_idx] = [(1.0, cod_state, terminal_reward, True)]
                    continue

                self.reset(seed)
                state = (pos_x, pos_y, rm_state)
                self.env.set_state(agent, state)
                for a_idx, action in enumerate(actions_list):
                    subactions, probabilities = self.env.get_action_distribution(action)

                    for subaction, prob in zip(subactions, probabilities):
                        self.reset(seed)
                        self.env.set_state(agent, state)  # Set current state

                        # Prepare a dictionary of actions for all agents
                        actions = {
                            a.name: (
                                subaction
                                if a.name == agent.name
                                else a.get_random_action()
                            )
                            for a in agents
                        }

                        try:
                            obs, rewards, terminations, truncations, infos = self.step(
                                actions
                            )
                        except Exception as e:
                            print(f"Error during `step` for agent {agent.name}: {e}")
                            continue

                        # Next state for the current agent
                        # next_state = self.env.get_current_state()
                        next_state_ag = agent.get_state()
                        next_state_rm = agent.get_reward_machine().get_current_state()
                        cod_next_state, _ = agent.encoder.encode(
                            next_state_ag, next_state_rm
                        )

                        # s_next_id = self.env.state_to_id(next_state)

                        # Determines whether the state is terminal
                        done = terminations[agent.name] or truncations[agent.name]

                        # Add transition to P matrix
                        reward = rewards[agent.name]
                        P[cod_state][a_idx].append((prob, cod_next_state, reward, done))

            all_P[agent.name] = P
            all_num_states[agent.name] = num_states
            all_num_actions[agent.name] = num_actions

            print(f"MDP construction completed for agent {agent.name}.")

        # Restore the initial state
        self.reset(seed)
        return all_P, all_num_states, all_num_actions
