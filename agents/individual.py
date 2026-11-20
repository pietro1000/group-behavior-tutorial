"""
Agent design 1 — the individual: one agent decides alone.
"""

from typing import List, Tuple

from core.agent import Agent
from core.conversation import Conversation


class IndividualAgent(Agent):
    """One model call per decision."""

    def run(self, *, conversation: Conversation, prompt: str, options: List[str]) -> Tuple[str, str]:
        """Send the prompt to the model and return its reply, as the reply and as the transcript."""
        # The prompt already names the options, so an individual has nothing to do with them.
        reply = self.client.generate(conversation=conversation, prompt=prompt)
        # One model call is all an individual does, so its reply is also its whole transcript.
        return reply, reply
