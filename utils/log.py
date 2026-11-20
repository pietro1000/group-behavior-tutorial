"""
The log: two text files per player per round.

logs/<date>_<time>/round_1/Alice_prompt.txt   what Alice read: the conversation so far, then the new prompt
                           Alice_answer.txt   what Alice produced: her transcript
                           Bob_prompt.txt
                           Bob_answer.txt

A group gets the same two files as an individual: the one prompt the game sent to the group, and
one answer file with every member's reply and the vote.
"""

from datetime import datetime
from pathlib import Path

from core.conversation import Conversation

# One folder per run, named by the time it started, so runs do not overwrite each other.
LOGS = Path("logs") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def log_prompt(*, round_number: int, name: str, conversation: Conversation, prompt: str) -> None:
    """Write what player `name` reads in this round: every message so far, then the new prompt.

    Each message is under its role: [user] is the game speaking, [assistant] is the player's own
    earlier reply. This is the list of messages the client sends to the model.
    """
    messages = conversation.messages + [{"role": "user", "content": prompt}]
    text = "\n\n".join(f"[{message['role']}]\n{message['content']}" for message in messages)
    _write(round_number=round_number, filename=f"{name}_prompt.txt", text=text)


def log_answer(*, round_number: int, name: str, answer: str) -> None:
    """Write what player `name` produced in this round: the transcript its run() returned."""
    _write(round_number=round_number, filename=f"{name}_answer.txt", text=answer)


def _write(*, round_number: int, filename: str, text: str) -> None:
    """Write `text` to logs/<date>_<time>/round_<round_number>/<filename>."""
    folder = LOGS / f"round_{round_number}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / filename).write_text(text + "\n")
