from multiagent_rlrm.multi_agent.event_detector import EventDetector


class PositionEventDetector(EventDetector):
    """
    Classe che rileva eventi basati sulla posizione corrente dell'agente.
    """

    def __init__(self, positions):
        """
        Inizializza il detector con le posizioni rilevanti.
        :param positions: Un insieme di tuple che rappresentano posizioni rilevanti.
        """
        self.positions = positions

    def detect_event(self, current_state):
        """
        Rileva un evento basato sulla posizione corrente dell'agente.
        :param current_state: Stato corrente dell'agente.
        :return: La posizione se l'evento è rilevato, altrimenti None.
        """
        ag_current_position = (current_state["pos_x"], current_state["pos_y"])
        if ag_current_position in self.positions:
            return ag_current_position
        return None
