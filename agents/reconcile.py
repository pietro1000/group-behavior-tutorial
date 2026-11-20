"""
Agent design 2 — ReConcile: a group of agents discusses, then votes.

Round 1: every member answers alone.
Later rounds: every member reads the other members' replies and answers again.
The group's answer is the majority vote of the last round.
"""

from collections import Counter
from typing import List, Tuple

from agents.individual import IndividualAgent
from core.agent import Agent
from core.conversation import Conversation
from core.parser import parse_answer

FOLLOWUP = (
    "{prompt}\n\n"
    "Other agents' replies from the previous round:\n"
    "{others}\n\n"
    "Carefully review these replies. Say which points you agree or disagree with and why, "
    "then give your own answer in the same format."
)


class ReconcileAgent(Agent):
    """A group that plays as one agent: same name + client + run() as an individual."""

    def __init__(self, *, name: str, client, n_members: int = 3, n_rounds: int = 2):
        super().__init__(name=name, client=client)
        self.members = [IndividualAgent(name=f"{name}_{i + 1}", client=client) for i in range(n_members)]
        self.n_rounds = n_rounds

    def run(self, *, conversation: Conversation, prompt: str, options: List[str]) -> Tuple[str, str]:
        """Discuss for n_rounds, then vote. The reply is the vote, the transcript the discussion."""
        transcript = []  # every member's reply in every discussion round, then the vote
        replies = {}
        for discussion_round in range(1, self.n_rounds + 1):
            new_replies = {}
            for member in self.members:
                # A member reads the others' previous replies, never its own.
                others = "\n".join(f"- {name}: {reply}" for name, reply in replies.items() if name != member.name)
                member_prompt = FOLLOWUP.format(prompt=prompt, others=others) if replies else prompt
                # A member is an individual, so its transcript is its reply: we keep the reply.
                reply, _ = member.run(conversation=conversation, prompt=member_prompt, options=options)
                entry = f"[{member.name}, discussion round {discussion_round}] {reply}"
                print(f"\n{entry}")
                transcript.append(entry)
                new_replies[member.name] = reply
            replies = new_replies

        # The group needs the options to count the votes.
        votes = Counter(parse_answer(reply, options) for reply in replies.values())
        answer = f"Answer: {votes.most_common(1)[0][0]}"
        print(f"\n[{self.name}] votes: {dict(votes)}")
        transcript.append(f"[{self.name}] votes: {dict(votes)}\n{answer}")
        return answer, "\n\n".join(transcript)
