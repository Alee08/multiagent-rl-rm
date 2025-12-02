class Message:
    """Represents a message sent by an agent with an associated condition."""

    def __init__(self, sender, condition):
        self.sender = sender  # The agent who sends the message
        self.condition = condition  # The condition that is being communicated
