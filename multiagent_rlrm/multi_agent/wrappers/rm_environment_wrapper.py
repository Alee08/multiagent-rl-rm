from multiagent_rlrm.learning_algorithms.qlearning import QLearning


class RMEnvironmentWrapper:
    """
    A wrapper around a multi-agent environment that integrates Reward Machines (RMs).

    This wrapper:
    - Resets RMs alongside the environment.
    - Updates RM state transitions and rewards at every step.
    - Augments the environment's step() output with RM-related information.
    - (Optionally) generates QRM experiences for QRMax-style learning.
    """

    def __init__(self, env, agents):
        """
        Initialize the RMEnvironmentWrapper.

        Args:
            env: The base multi-agent environment being wrapped.
            agents (list): A list of RL-enabled agents interacting with the environment.
        """
        self.env = env
        self.agents = agents
        # Keeps RM reward near 1 depending on the optimal RM trace length
        self.reward_modifier = 1

    def reset(self, seed):
        """
        Reset the environment and all Reward Machines to their initial states.

        Args:
            seed: Environment reset seed.

        Returns:
            (observations, infos): Two dictionaries matching the environment API.
        """
        observations, infos = self.env.reset(seed)
        for agent in self.agents:
            agent.get_reward_machine().reset_to_initial_state()
        return observations, infos

    def step(self, actions):
        """
        Perform one environment step and update Reward Machine states.

        Args:
            actions (dict): Maps agent_name → action

        Returns:
            observations, rewards, terminations, truncations, infos
        """
        observations, rewards, env_terminations, env_truncations, infos = self.env.step(
            actions
        )

        for agent in self.agents:
            rm = agent.get_reward_machine()

            current_state = infos[agent.name].get("prev_s", observations[agent.name])
            next_state = observations[agent.name]

            # RM state transition
            state_rm = rm.get_current_state()
            reward_rm = rm.step(next_state)
            new_state_rm = rm.get_current_state()

            # Optional scaling
            reward_rm *= self.reward_modifier

            # Add RM info into infos
            infos[agent.name]["RQ"] = reward_rm
            infos[agent.name]["prev_q"] = state_rm
            infos[agent.name]["q"] = new_state_rm
            infos[agent.name]["reward_machine"] = rm

            # QRM (QRMax) auxiliary experiences
            use_qrm = getattr(agent.get_learning_algorithm(), "use_qrm", False)
            if use_qrm:
                qrm_experiences = self._get_qrm_experiences(
                    agent,
                    current_state,
                    next_state,
                    actions[agent.name],
                    rewards[agent.name],
                    new_state_rm,
                    env_terminations[agent.name],
                )
                infos[agent.name]["qrm_experience"] = qrm_experiences

            # Add RM reward to the environment reward
            rewards[agent.name] += reward_rm

        # Combine environment termination with RM termination
        rm_terminations = self.check_terminations()
        terminations = {
            agent.name: (env_terminations[agent.name] or rm_terminations[agent.name])
            for agent in self.agents
        }

        for agent in self.agents:
            infos[agent.name]["env_terminated"] = env_terminations[agent.name]
            infos[agent.name]["rm_terminated"] = rm_terminations[agent.name]

        truncations = env_truncations

        return observations, rewards, terminations, truncations, infos

    def check_terminations(self):
        """
        Check which agents have reached the final RM state.

        Returns:
            rm_terminations (dict): agent_name → bool
        """
        rm_terminations = {}
        for agent in self.agents:
            rm = agent.get_reward_machine()
            rm_terminations[agent.name] = rm.get_current_state() == rm.get_final_state()
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
        """
        Generate experiences of the form used by QRMax/QRM.

        These simulate transitions for *all possible RM states*, not only the current one.

        Returns:
            List of QRM experience tuples.
        """
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

            # If transition does not exist, RM stays where it is
            if hypothetical_next_state is None:
                hypothetical_next_state = state_rm

            encoded_state, enc_state_info = agent.encoder.encode(
                current_state, state_rm
            )
            encoded_next_state, enc_next_state_info = agent.encoder.encode(
                next_state, hypothetical_next_state
            )

            rm_done = hypothetical_next_state == final_state

            qrm_experience = (
                encoded_state,
                action_index,
                env_reward + hypothetical_reward,
                encoded_next_state,
                env_termination or rm_done,
                enc_state_info["s"],
                enc_state_info["q"],
                enc_next_state_info["s"],
                enc_next_state_info["q"],
                hypothetical_reward,
            )

            qrm_experiences.append(qrm_experience)

        return qrm_experiences

    def get_mdp(self, seed):
        """
        Construct the full MDP (transition model P) induced by the environment and each agent's Reward Machine.

        This is used for algorithms requiring full transition models (e.g., RMax, QRMax).

        Returns:
            all_P: dict(agent_name → transition dict P)
            all_num_states: dict(agent_name → number of encoded states)
            all_num_actions: dict(agent_name → number of actions)
        """
        self.env.stochastic = False
        width = self.env.map_width
        height = self.env.map_height
        agents = self.env.agents

        all_P = {}
        all_num_states = {}
        all_num_actions = {}

        for agent in agents:
            rm = agent.get_reward_machine()
            num_rm_states = rm.numbers_state()
            num_states = width * height * num_rm_states

            actions_list = agent.get_actions()
            num_actions = len(actions_list)

            P = {
                s_id: {a_idx: [] for a_idx in range(num_actions)}
                for s_id in range(num_states)
            }

            for cod_state in range(num_states):

                decoded_state, rm_state = agent.encoder.decode(cod_state)
                pos_x, pos_y = decoded_state["pos_x"], decoded_state["pos_y"]

                is_terminal, terminal_reward = self.env.is_terminal_state_mdp(
                    agent, pos_x, pos_y, rm_state
                )

                if is_terminal:
                    # Terminal state self-loops
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
                        self.env.set_state(agent, state)

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
                        except Exception:
                            continue

                        next_state_ag = agent.get_state()
                        next_state_rm = agent.get_reward_machine().get_current_state()

                        cod_next_state, _ = agent.encoder.encode(
                            next_state_ag, next_state_rm
                        )

                        done = terminations[agent.name] or truncations[agent.name]
                        reward = rewards[agent.name]

                        P[cod_state][a_idx].append((prob, cod_next_state, reward, done))

            all_P[agent.name] = P
            all_num_states[agent.name] = num_states
            all_num_actions[agent.name] = num_actions

        self.reset(seed)
        return all_P, all_num_states, all_num_actions
