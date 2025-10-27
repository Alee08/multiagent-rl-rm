from unified_planning.model.multi_agent import Agent
from multiagent_rlrm.multi_agent.reward_machine import RewardMachine
from typing import Optional
from multiagent_rlrm.utils.message import Message
from multiagent_rlrm.multi_agent.state_encoder import StateEncoder


class AgentRL(Agent):
    """
    Agent
    """

    def __init__(
        self, name: str, ma_problem, reward_machine: Optional["RewardMachine"] = None
    ):
        """
        Inizializza un agente con le capacità di RL.

        :param name: Nome univoco dell'agente.
        :param ma_problem: Riferimento al problema di planning multi-agente.
        :param reward_machine: Istanza di RewardMachine specifica per questo agente.
        """
        super().__init__(name, ma_problem)
        self.reward_machine = reward_machine
        self.ma_problem = ma_problem
        self.actions_dict = {}
        self.learning_algorithm = None
        self.message_conditions = None
        self.messages = {}  # Uno stato interno per tracciare informazioni rilevanti
        self.message_sent = False
        self.position = None
        self.state = {}
        self.actions_ = []
        self.initial_position = {}
        self.initial_state = {}
        self.rm_state = None
        self.next_rm_state = None
        self.encoder = None

    def action(self, name: str) -> "up.model.action.Action":
        """
        Returns the `action` with the given `name`.

        :param name: The `name` of the target `action`.
        :return: The `action` in the `problem` with the given `name`.
        """
        for a in self.actions_:
            if a.name == name:
                return a
        raise UPValueError(f"Action of name: {name} is not defined!")

    def add_action(self, action):
        self.actions_.append(action)

    def get_actions(self):
        return self.actions_

        # Potrebbe essere necessario aggiungere ulteriori attributi specifici per il RL

    def add_state_encoder(self, encoder):
        """
        Imposta o sostituisce l'encoder dello stato dell'agente.
        :param encoder: Istanza dell'encoder da impostare.
        """
        self.encoder = encoder

    def select_action(self, state, best=False):
        """
        Seleziona l'azione da eseguire in base allo stato corrente e allo stato della Reward Machine.

        :param state: Stato corrente dell'agente.
        :param rm_state: Stato corrente della Reward Machine.
        :param best: Se True, seleziona l'azione in modalità best (e.g., no epsilon-greedy).
        :return: L'azione selezionata.
        """
        if not self.encoder:
            raise Exception(
                "Encoder not set. Please add an encoder before selecting actions."
            )
        encoded_state, info = self.encoder.encode(state)

        # Seleziona l'indice dell'azione dall'algoritmo di apprendimento
        action_index = self.get_learning_algorithm().choose_action(
            encoded_state, best, info=info
        )

        # Recupera l'azione corrispondente all'indice e la restituisce
        action = self.actions_dix()[action_index]
        if action is None:
            raise ValueError(
                f"Action index {action_index} not found in actions dictionary."
            )
        return action

    def update_policy(self, state, action, reward, next_state, terminated, **kwargs):
        # Ensure necessary info is provided, otherwise use default values
        infos = kwargs.get("infos", {})

        # Ensure encoder is set
        if not self.encoder:
            raise Exception(
                "Encoder not set. Please add an encoder before updating policy."
            )
        # Ensure RM is set
        if not self.reward_machine:
            raise Exception(
                "Reward Machine not set. Cannot update policy without Reward Machine."
            )

        # Encode the current and next state using the encoder
        # encoded_current_state, current_info = self.encoder.encode(state)
        # encoded_next_state, next_info = self.encoder.encode(next_state)

        # Use values from `infos` or use those from the encoder if not specified
        # prev_s = infos.get("prev_s", 0)  # Default to 0 if not provided
        state_rm = infos.get("prev_q", 0)  # Default to 0 if not provided
        next_state_rm = infos.get("q", 0)  # Default to 0 if not provided
        reward_env = infos.get("Renv", 0)  # Default to 0 if not provided
        reward_q = infos.get("RQ", 0)  # Default to 0 if not provided
        qrm_experience = infos.get("qrm_experience", [])  # Default to 0 if not provided
        reward_machine = infos.get("reward_machine", [])

        encoded_current_state, current_info = self.encoder.encode(state, state_rm)
        encoded_next_state, next_info = self.encoder.encode(next_state, next_state_rm)
        # print("Agent: ", state, next_state, encoded_current_state, state_rm, encoded_next_state, next_state_rm, "weeeeeeeeeeeeee")
        # Convert Reward Machine states to indices if they come from infos
        state_rm_idx = (
            self.reward_machine.get_state_index(state_rm) if state_rm != 0 else 0
        )
        next_state_rm_idx = (
            self.reward_machine.get_state_index(next_state_rm)
            if next_state_rm != 0
            else 0
        )

        # Prepare complete information
        info = {
            "prev_s": current_info["s"],
            "s": next_info["s"],
            "prev_q": state_rm_idx,
            "q": next_state_rm_idx,
            "Renv": reward_env,
            "RQ": reward_q,
            "qrm_experience": qrm_experience,
            "reward_machine": reward_machine,
        }
        # print(info["prev_s"], info["s"], info["prev_q"], info["q"], info["Renv"], info["RQ"])

        action_index = self.actions_idx(action)
        # Update learning algorithm
        term = self.get_learning_algorithm().update(
            encoded_current_state,
            encoded_next_state,
            action_index,
            reward,
            terminated,
            info=info,
        )

        return term  # whether the alg ask to terminate

    def actions_dix(self):
        for idx, act in enumerate(self.actions_):
            self.actions_dict[idx] = act
        return self.actions_dict

    def actions_idx(self, action):
        dict = self.actions_dix()
        # Trovare la chiave per un valore specifico
        chiave_trovata = None
        for chiave, valore in dict.items():
            if valore == action:
                chiave_trovata = chiave
                break

        return chiave_trovata

    def get_reward(self, event):
        return self.reward_machine.get_reward(event) if self.reward_machine else 0

    def set_reward_machine(self, reward_machine: RewardMachine):
        self.reward_machine = reward_machine

    def get_reward_machine(self):
        return self.reward_machine

    def add_rl_action(self, action):
        """
        Aggiunge un'azione specifica per il RL all'agente.

        :param action: Azione RL da aggiungere.
        """
        # Implementazione dipende dalla struttura delle azioni RL
        self.actions_.append(action)

    # Altri metodi specifici per il RL possono essere aggiunti qui

    def set_learning_algorithm(self, algorithm):
        """
        Assegna un algoritmo di apprendimento all'agente.

        :param algorithm: Istanza dell'algoritmo di apprendimento.
        """
        self.learning_algorithm = algorithm

    def get_learning_algorithm(self):
        """
        Ritorna l'algoritmo di apprendimento dell'agente.
        """
        return self.learning_algorithm

    def _send_message(self, agents, condition):
        """
        Invia un messaggio agli altri agenti quando l'agente raggiunge uno stato "X".

        :param condition: La condizione da comunicare agli altri agenti.
        """
        message = Message(self.name, condition)
        # Logica per inviare il messaggio agli altri agenti
        # Potrebbe essere qualcosa come:
        self.ma_problem.broadcast_message(agents, message)

    def _receive_message(self, message):
        # Gestisci il messaggio ricevuto
        if isinstance(message, Message):
            # Aggiorna lo stato interno in base al messaggio
            self.process_message(message)

    def process_message(self, message):
        # Logica specifica per processare il messaggio
        chiave_messaggio = (
            message.sender,
            message.condition[0][0],
        )  # Crea una tupla (agente, fluent)
        self.messages[chiave_messaggio] = message.condition[0][1]
        # self.messages[message.sender] = message.condition

        # Logica per certi tipi di messaggi
        """if message.condition == some_specific_condition:
            self.take_specific_action()"""

    def take_specific_action(self):
        # Implementa azioni specifiche da intraprendere in risposta a certi messaggi o condizioni
        pass

    def reset_messages(self):
        self.messages = {}
        self.message_conditions = None

    def return_messages(self):
        return self.messages

    def execute_action(self, action):
        if action is None:
            raise ValueError(
                f"Azione {action.name} non trovata per l'agente {self.name}."
            )

        if all(pre(self) for pre in action.preconditions):
            for effect in action.effects:
                effect(self)
            return True
        else:
            # print(f"Precondizioni per l'azione {action.name} non soddisfatte.")
            return False

    def set_initial_position(self, pos_x, pos_y):
        """Memorizza e imposta la posizione iniziale dell'agente."""
        self.initial_position = (pos_x, pos_y)
        self.set_position(pos_x, pos_y)

    def set_position(self, pos_x, pos_y):
        """Imposta la posizione dell'agente e aggiorna lo stato."""
        self.position = (pos_x, pos_y)
        self.add_to_state("pos_x", pos_x)
        self.add_to_state("pos_y", pos_y)

    def get_position(self):
        return self.position

    def add_to_state(self, key, value):
        """Aggiunge/aggiorna un attributo nello stato e nello stato iniziale."""
        self.state[key] = value
        self.initial_state[key] = value  # Memorizza anche nello stato iniziale

    def reset(self):
        """Reimposta la posizione e lo stato dell'agente alla configurazione iniziale."""
        self.set_position(
            *self.initial_position
        )  # Utilizza la posizione iniziale memorizzata
        self.state = (
            self.initial_state.copy()
        )  # Resetta lo stato agli attributi iniziali
        self.reset_messages()  # Resetta eventuali messaggi
        if self.reward_machine:
            self.reward_machine.reset_to_initial_state()  # Resetta la Reward Machine se presente

    def set_state(self, **kwargs):
        """
        Imposta gli attributi dello stato dell'agente.

        :param kwargs: Dizionario degli attributi dello stato con i rispettivi valori.
        """
        for key, value in kwargs.items():
            self.state[key] = value

    def get_state(self):
        """
        Ritorna lo stato corrente dell'agente.

        :return: Dizionario che rappresenta lo stato dell'agente.
        """
        return self.state
