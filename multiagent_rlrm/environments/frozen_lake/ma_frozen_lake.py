import numpy as np
import random
from multiagent_rlrm.utils.utils import encode_state
from multiagent_rlrm.learning_algorithms.qlearning_lambda import QLearningLambda
from multiagent_rlrm.learning_algorithms.qlearning import QLearning
from multiagent_rlrm.multi_agent.base_environment import BaseEnvironment

# from building_RM import RM_dict, RM_dict_true, RM_dict_true_seq
random.seed(a=123)
np.random.seed(123)


class MultiAgentFrozenLake(BaseEnvironment):
    metadata = {"name": "multi_agent_frozen_lake"}

    def __init__(self, width, height, holes):
        # Passa i valori di epsilon al costruttore della classe base
        super().__init__(width, height)
        self.holes = holes  # Liste delle posizioni dei buchi
        self.possible_actions = ["up", "down", "left", "right"]
        self.rewards = 0
        self.frozen_lake = False
        self.penalty_amount = 0
        self.active_agents = {agent.name: True for agent in self.agents}
        self.agent_fail = {agent.name: False for agent in self.agents}
        self.agent_steps = {agent.name: 0 for agent in self.agents}
        self.delay_action = False  # Aggiunto attributo per il delay delle azioni

    def reset(self, seed=123, options=None):
        """Reset set the environment to a starting point."""
        self.rewards = {agent.name: 0 for agent in self.agents}
        self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)
        self.timestep = 0
        # self.agent_states = {agent.name: {} for agent in self.agents}
        self.active_agents = {agent.name: True for agent in self.agents}
        self.agent_fail = {agent.name: False for agent in self.agents}
        self.agent_steps = {agent.name: 0 for agent in self.agents}

        for agent in self.agents:
            agent.reset()  # reset della RM, messaggi e state dell'agente
            l_algo = agent.get_learning_algorithm()
            if isinstance(l_algo, QLearningLambda):
                l_algo.reset_e_table()  # Se è ql_lambda
            initial_position = (
                agent.get_state()
            )  # Assumi che get_state ritorni un dizionario con pos_x e pos_y

            if isinstance(l_algo, QLearning):
                l_algo.learn_done_episode()
            # self.agent_states[agent.name] = initial_position
            # agent.get_learning_algorithm().update_epsilon()
            # l_algo.learn_init()
            # l_algo.learn_init_episode()
            # l_algo.learn_done_episode()

        # observations = self.agent_states
        # Get dummy infos
        infos = {agent: {} for agent in self.agents}
        observations = {agent.name: agent.state for agent in self.agents}

        return observations, infos

    def step(self, actions):
        self.rewards = {a.name: 0 for a in self.agents}
        infos = {a.name: {} for a in self.agents}
        for agent in self.agents:
            if not self.active_agents[agent.name]:
                continue
            current_statee = self.get_state(agent)

            ag_action = actions[agent.name]  # azione scelta dall'agente
            # learning_algorithm = agent.get_learning_algorithm()

            if self.frozen_lake:
                action = self.get_stochastic_action(agent, ag_action.name)
                if action != "wait":
                    agent.execute_action(action)
                else:
                    pass
            else:
                agent.execute_action(ag_action)

            new_state = self.get_state(agent)
            # state_rm = agent.reward_machine.get_current_state()

            # Calcola la penalità se l'agente finisce su un buco
            reward_env = self.holes_in_the_ice(new_state, agent.name)

            # reward_rm = agent.get_reward_machine().step(new_state)  # Chiamata alla RM per ottenere la ricompensa
            # reward_rm = agent.get_reward_machine().get_reward(event)
            # reward = reward_rm + reward_env
            reward = reward_env
            # new_state_rm = agent.reward_machine.get_current_state()

            self.rewards[agent.name] += reward
            # Memorizza i due stati della RM per questo agente per l'uso nell'aggiornamento della politica
            # agent.rm_state = state_rm
            # agent.next_rm_state = new_state_rm

            # Incrementa il conteggio dei passi per l'agente attivo
            self.agent_steps[agent.name] += 1
            # Aggiorna `infos` con gli stati precedenti e correnti
            infos[agent.name]["prev_s"] = current_statee
            # infos[agent.name]["prev_q"] = state_rm
            infos[agent.name]["s"] = new_state
            # infos[agent.name]["q"] = new_state_rm
            infos[agent.name]["Renv"] = reward_env
            # infos[agent.name]["RQ"] = reward_rm
        self.timestep += 1
        # Aggiorna le informazioni di termination e troncamento per tutti gli agenti
        terminations, truncations = self.check_terminations()

        # Update active_agents based on terminations and truncations
        for agent_name in terminations:
            if terminations[agent_name]:
                self.active_agents[agent_name] = False

        # Restituisci le osservazioni, le ricompense, le terminazioni, i troncamenti e le informazioni aggiornate
        observations = {agent.name: agent.state for agent in self.agents}
        # print("aaa", observations, self.rewards, terminations, truncations, infos)
        return observations, self.rewards, terminations, truncations, infos

    def holes_in_the_ice(self, state, agent_name):
        agent_pos = (state["pos_x"], state["pos_y"])
        if agent_pos in self.holes:
            self.agent_fail[agent_name] = True  # Usa il nome dell'agente come chiave
            return self.penalty_amount
        else:
            return 0

    def check_terminations(self):
        terminations = {a.name: False for a in self.agents}
        truncations = {a.name: False for a in self.agents}

        for agente in self.agents:
            if self.agent_fail[agente.name] or self.timestep > 1000:
                terminations[agente.name] = True
                truncations[agente.name] = True

        return terminations, truncations

    def get_state(self, agent):
        current_state = agent.get_state()
        # Restituisce una copia dello stato corrente dell'agente per evitare modifiche accidentali
        return current_state.copy()

    def get_stochastic_action(self, agent, intended_action):
        if self.delay_action:
            action_map = {
                "left": (["wait", "left", "up", "down"], [0.6, 0.36, 0.02, 0.02]),
                "right": (["wait", "right", "up", "down"], [0.6, 0.36, 0.02, 0.02]),
                "up": (["wait", "up", "left", "right"], [0.6, 0.36, 0.02, 0.02]),
                "down": (["wait", "down", "left", "right"], [0.6, 0.36, 0.02, 0.02]),
            }
        else:
            action_map = {
                "left": (["left", "up", "down"], [0.8, 0.1, 0.1]),
                "right": (["right", "up", "down"], [0.8, 0.1, 0.1]),
                "up": (["up", "left", "right"], [0.8, 0.1, 0.1]),
                "down": (["down", "left", "right"], [0.8, 0.1, 0.1]),
            }
        actions, probabilities = action_map[intended_action]
        chosen_action = np.random.choice(actions, p=probabilities)
        if chosen_action != "wait":
            return agent.action(chosen_action)
        else:
            return "wait"
