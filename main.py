"""
Run the experiment: a tool-using agent plays an iterated prisoner's dilemma against a group.

Run: uv run main.py
"""

import platform

from agents.reconcile import ReconcileAgent
from agents.tool_use import ToolAgent
from experiment.prisoners_dilemma import PrisonersDilemma

# Two clients do the same job. Apple Silicon Macs get the faster one.
if platform.system() == "Darwin" and platform.machine() == "arm64":
    from clients.client_mlx import Client
else:
    from clients.client import Client

N_ROUNDS = 5

if __name__ == "__main__":
    client = Client()  # one model, shared by every agent

    alice = ToolAgent(name="Alice", client=client)
    bob = ReconcileAgent(name="Bob", client=client, n_members=3, n_rounds=2)

    game = PrisonersDilemma(players=[alice, bob], n_rounds=N_ROUNDS)
    totals = game.run()
    print(f"\nTotal points: {totals}")
