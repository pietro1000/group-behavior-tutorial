"""
The agent: a name, a client and one method, run().

run() is how the game asks an agent to decide. An agent design (the files in agents/) is a way to
implement it: one model call, a group discussion, a tool call followed by a model call.
"""

from typing import List, Tuple

from core.conversation import Conversation


class Agent:
    """Base class of every agent design."""

    def __init__(self, *, name: str, client):
        self.name = name
        self.client = client  # the Client of clients/client.py or of clients/client_mlx.py

    def run(self, *, conversation: Conversation, prompt: str, options: List[str]) -> Tuple[str, str]:
        """Decide: reply to `prompt` with one of `options`, given the conversation so far.

        Returns two texts:
        - the reply, which starts with "Answer: <one of options>". The game reads the move from it
          and adds it to the conversation, so it is what the agent remembers in later rounds.
        - the transcript, everything the agent produced on the way to that reply. The game writes
          it to logs/ and nothing else reads it.

        For an individual the two are the same text. For a group the reply is the vote and the
        transcript is the whole discussion.
        """
        raise NotImplementedError("Each agent design implements run().")
