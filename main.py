"""
Run the experiment: an individual agent plays an iterated prisoner's dilemma against a group.

Run: uv run main.py
"""

import platform

from individual import IndividualAgent
from prisoners_dilemma import PrisonersDilemma
from reconcile import ReconcileAgent

# Two clients do the same job. Apple Silicon Macs get the faster one.
if platform.system() == "Darwin" and platform.machine() == "arm64":
    from client_mlx import Client
else:
    from client import Client

N_ROUNDS = 5

if __name__ == "__main__":
    client = Client()  # one model, shared by every agent

    alice = IndividualAgent(name="Alice", client=client)
    bob = ReconcileAgent(name="Bob", client=client, n_members=3, n_rounds=2)

    game = PrisonersDilemma(players=[alice, bob], n_rounds=N_ROUNDS)
    totals = game.run()
    print(f"\nTotal points: {totals}")
