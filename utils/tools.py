"""
The tools: the Python functions an agent can call.

count_moves counts what the other player has done so far. Counting is the kind of thing a small
model gets wrong and Python always gets right, which is the reason to give an agent a tool at all.
"""

import re

from core.conversation import Conversation


def count_moves(*, conversation: Conversation, action: str) -> str:
    """How many times the other player has chosen `action` in the rounds played so far."""
    # The game speaks as "user", and its round results read ", the other player chose DEFECT."
    # (FEEDBACK in experiment/prisoners_dilemma.py: change that wording and this pattern must follow.)
    said = "\n".join(m["content"] for m in conversation.messages if m["role"] == "user")
    moves = re.findall(r"the other player chose (\w+)", said)
    if not moves:
        return "No rounds have been played yet, so the other player has made no choices."
    return (
        f"In the {len(moves)} rounds played so far, "
        f"the other player chose {action} {moves.count(action)} times."
    )
