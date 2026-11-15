"""
Agent design 1 — the individual: one agent decides alone.
"""

from typing import List

from core.agent import Agent
from core.conversation import Conversation


class IndividualAgent(Agent):
    """One model call per decision."""

    def run(self, *, conversation: Conversation, prompt: str, options: List[str]) -> str:
        # The prompt already names the options, so an individual has nothing to do with them.
        return self.client.generate(conversation=conversation, prompt=prompt)
