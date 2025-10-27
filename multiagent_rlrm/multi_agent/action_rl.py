class ActionRL:
    def __init__(self, name, preconditions=None, effects=None):
        """
        Inizializza un'azione RL.

        :param name: Nome dell'azione.
        :param environment: Riferimento all'ambiente in cui l'azione verrà eseguita.
        :param preconditions: Lista di precondizioni per l'azione.
        :param effects: Lista di effetti dell'azione.
        """
        self.name = name
        self.preconditions = (
            preconditions
            if isinstance(preconditions, (list, tuple))
            else [preconditions]
        )
        self.effects = effects if isinstance(effects, (list, tuple)) else [effects]
