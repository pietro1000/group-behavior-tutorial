"""
The agent: a name, a client and one method, run().

An agent design (individual.py, reconcile.py) is a way to implement run().
"""

from typing import List

from core.conversation import Conversation


class Agent:
    """Base class of every agent design."""

    def __init__(self, *, name: str, client):
        self.name = name
        self.client = client  # the Client of client.py or of client_mlx.py

    def run(self, *, conversation: Conversation, prompt: str, options: List[str]) -> str:
        """Reply to `prompt` with one of `options`, given the conversation so far."""
        raise NotImplementedError("Each agent design implements run().")
