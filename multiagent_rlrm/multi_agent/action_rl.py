class ActionRL:
    """Lightweight representation of an RL action with optional preconditions and effects."""

    def __init__(self, name, preconditions=None, effects=None):
        """
        Initialize an RL action.

        :param name: Name of the action.
        :param preconditions: List of preconditions for the action.
        :param effects: List of effects of the action.
        """
        self.name = name

        # Optional preconditions
        if preconditions is None:
            self.preconditions = []
        elif isinstance(preconditions, (list, tuple)):
            self.preconditions = list(preconditions)
        else:
            self.preconditions = [preconditions]

        # Optional effects
        if effects is None:
            self.effects = []
        elif isinstance(effects, (list, tuple)):
            self.effects = list(effects)
        else:
            self.effects = [effects]
