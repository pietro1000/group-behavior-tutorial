```
 ██████╗ ██████╗  ██████╗ ██╗   ██╗██████╗
██╔════╝ ██╔══██╗██╔═══██╗██║   ██║██╔══██╗
██║  ███╗██████╔╝██║   ██║██║   ██║██████╔╝
██║   ██║██╔══██╗██║   ██║██║   ██║██╔═══╝
╚██████╔╝██║  ██║╚██████╔╝╚██████╔╝██║
 ╚═════╝ ╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═╝
██████╗ ███████╗██╗  ██╗ █████╗ ██╗   ██╗██╗ ██████╗ ██████╗
██╔══██╗██╔════╝██║  ██║██╔══██╗██║   ██║██║██╔═══██╗██╔══██╗
██████╔╝█████╗  ███████║███████║██║   ██║██║██║   ██║██████╔╝
██╔══██╗██╔══╝  ██╔══██║██╔══██║╚██╗ ██╔╝██║██║   ██║██╔══██╗
██████╔╝███████╗██║  ██║██║  ██║ ╚████╔╝ ██║╚██████╔╝██║  ██║
╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝
          of AI — a hands-on tutorial
```

# Group behavior of AI — tutorial

Does a **group** of AI agents behave differently from a **single** AI agent?

This repository is the smallest experiment that lets you look: an *individual* agent (Alice)
plays an **iterated prisoner's dilemma** against a *group* of three agents (Bob). The group
discusses and votes before every move, following ReConcile (Chen, Saha & Bansal, 2024 ACL).
Everything runs on your laptop with a small open model — no API key, no cloud.

## Layout

```
group-behavior-tutorial/
├── core/
│   ├── conversation.py     # Conversation: the list of messages the model reads
│   ├── agent.py            # Agent: a name, a client and one method, run()
│   └── parser.py           # parse_answer(): reads "Answer: COOPERATE" out of a reply
├── client.py               # talks to the model: PyTorch, runs on any machine
├── client_mlx.py           # the same client for Apple Silicon Macs: MLX, faster
├── individual.py           # agent design 1: one agent decides alone
├── reconcile.py            # agent design 2: a group discusses, then votes
├── prisoners_dilemma.py    # the experiment: payoffs, prompts, the loop over rounds
├── main.py                 # builds the two players and runs the game
└── pyproject.toml  uv.lock # the Python environment (uv)
```

About 300 lines in total. A good reading order is the one above, top to bottom.

## Setup (uv)

1. Install [uv](https://docs.astral.sh/uv/), the Python environment manager:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh                                         # macOS / Linux
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"      # Windows
```

2. Download the code and build the environment (uv also installs Python 3.12 if you lack it):

```bash
git clone https://github.com/pietro1000/group-behavior-tutorial.git
cd group-behavior-tutorial
uv sync
```

## Running

```bash
uv run main.py
```

The model is DeepSeek-R1-Distill-Qwen-1.5B. `main.py` picks the client for your machine:

| Your machine | Client | First-run download | One game (5 rounds, 35 model calls) |
|---|---|---|---|
| Apple Silicon Mac | `client_mlx.py` (MLX, 4-bit model) | about 1 GB | 13 s on an M3 Ultra |
| Anything else | `client.py` (PyTorch) | about 3.5 GB | 3 min on the same Mac's CPU; faster with an NVIDIA GPU |

The **first run downloads the model**, so run it once before the session, on a good connection.
A laptop without a GPU is slower than the numbers above: if a game takes too long, lower
`N_ROUNDS` in `main.py`.

You see every reply as it is produced: Alice's move, each group member's reply in the two
discussion rounds, the group's vote, and the points.

```
=============== Round 1 ===============

[Alice] Answer: DEFECT
Reasoning: ...

[Bob_1, discussion round 1] Answer: COOPERATE
Reasoning: ...
...
[Bob_3, discussion round 2] Answer: DEFECT
Reasoning: ...

[Bob] votes: {'DEFECT': 3}

[Bob] Answer: DEFECT

-> Alice chose DEFECT: 1 points (total 1)
-> Bob chose DEFECT: 1 points (total 1)
```

**A note on the model.** This model normally writes a long hidden reasoning before every reply,
which takes minutes per reply on a laptop. To keep the tutorial fast, the client skips that
reasoning and makes the model state its answer first, then explain it. The model is tiny, so
expect its explanations to be shaky: the point here is the *mechanics* of agents and groups.

## Things to try

All in `main.py` unless noted:

- **Swap the designs** — make both players individuals, or both groups.
- **Change the group** — `n_members=5`, or `n_rounds=1` (no discussion: a plain majority vote).
- **Play longer** — raise `N_ROUNDS`.
- **Change the game** — edit `PAYOFFS` and `INSTRUCTIONS` in `prisoners_dilemma.py` (keep them in line).
- **Change the randomness** — `Client(temperature=1.0)`.
