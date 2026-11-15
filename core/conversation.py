"""
The conversation with one player: the list of messages the model reads.

The game owns the conversation. An agent has no memory of its own.
"""

from typing import Dict, List


class Conversation:
    """A list of {"role", "content"} messages, the format every chat model expects."""

    def __init__(self):
        self.messages: List[Dict[str, str]] = []

    def add(self, *, role: str, content: str) -> None:
        """Append one message. The role is "user" (the game speaks) or "assistant" (the agent replied)."""
        self.messages.append({"role": role, "content": content})
