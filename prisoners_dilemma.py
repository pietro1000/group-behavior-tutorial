"""
The experiment — an iterated prisoner's dilemma between two agents.

Each round both players choose COOPERATE or DEFECT without seeing the other's choice.
The game keeps one conversation per player and adds the result of every round to it.
"""

from typing import Dict, List

from core.agent import Agent
from core.conversation import Conversation
from core.parser import parse_answer

ACTIONS = ["COOPERATE", "DEFECT"]

# (my action, the other's action) -> my points
PAYOFFS = {
    ("COOPERATE", "COOPERATE"): 3,
    ("COOPERATE", "DEFECT"): 0,
    ("DEFECT", "COOPERATE"): 5,
    ("DEFECT", "DEFECT"): 1,
}

# The points in the text must match PAYOFFS.
INSTRUCTIONS = """You are a player in a game with one other player. The game lasts {n_rounds} rounds.
In each round, you and the other player each choose COOPERATE or DEFECT, without seeing the other's choice.
- If both choose COOPERATE, each earns 3 points.
- If both choose DEFECT, each earns 1 point.
- If one chooses DEFECT and the other chooses COOPERATE, the one who chose DEFECT earns 5 points and the other earns 0 points.
Your goal is to earn as many points as possible over the {n_rounds} rounds."""

QUESTION = """Round {round_number}: do you choose COOPERATE or DEFECT?
Reply in this format:
Answer: <COOPERATE or DEFECT>
Reasoning: <one sentence>"""

FEEDBACK = "Result of round {round_number}: you chose {mine}, the other player chose {theirs}. You earned {points} points."


class PrisonersDilemma:
    """Two players, n_rounds rounds."""

    def __init__(self, *, players: List[Agent], n_rounds: int):
        self.players = players
        self.n_rounds = n_rounds
        # One conversation per player, starting with the rules of the game.
        self.conversations = {player.name: Conversation() for player in players}
        for conversation in self.conversations.values():
            conversation.add(role="user", content=INSTRUCTIONS.format(n_rounds=n_rounds))

    def run(self) -> Dict[str, int]:
        """Play all rounds and return the total points of each player."""
        totals = {player.name: 0 for player in self.players}

        for round_number in range(1, self.n_rounds + 1):
            print(f"\n=============== Round {round_number} ===============")
            question = QUESTION.format(round_number=round_number)

            # 1. Each player decides, reading only its own conversation.
            actions = []
            for player in self.players:
                conversation = self.conversations[player.name]
                reply = player.run(conversation=conversation, prompt=question, options=ACTIONS)
                print(f"\n[{player.name}] {reply}")
                conversation.add(role="user", content=question)
                conversation.add(role="assistant", content=reply)
                action = parse_answer(reply, ACTIONS)
                if action is None:
                    print(f"[{player.name}] no valid answer: counted as DEFECT")
                    action = "DEFECT"
                actions.append(action)

            # 2. The game scores the round and tells each player what happened.
            print()
            for player, mine, theirs in zip(self.players, actions, reversed(actions)):
                points = PAYOFFS[(mine, theirs)]
                totals[player.name] += points
                feedback = FEEDBACK.format(round_number=round_number, mine=mine, theirs=theirs, points=points)
                self.conversations[player.name].add(role="user", content=feedback)
                print(f"-> {player.name} chose {mine}: {points} points (total {totals[player.name]})")

        return totals
