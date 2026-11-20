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

This repository is the smallest experiment that lets you look: a single agent (Alice) plays an
**iterated prisoner's dilemma** against a *group* of three agents (Bob). The group discusses and
votes before every move, following ReConcile (Chen, Saha & Bansal, 2024 ACL). Alice decides alone,
but she can **call a Python function** to look up what Bob has done so far.
Everything runs on your laptop with a small open model — no API key, no cloud.

## Layout

```
group-behavior-tutorial/
├── core/                       # what every agent is made of
│   ├── conversation.py         # Conversation: the list of messages the model reads
│   ├── agent.py                # Agent: a name, a client and one method, run()
│   └── parser.py               # parse_answer(): reads "Answer: COOPERATE" out of a reply
├── clients/                    # two ways to run the model, same interface
│   ├── client.py               # PyTorch: runs on any machine
│   └── client_mlx.py           # MLX: the same job on Apple Silicon Macs, faster
├── agents/                     # the three designs this tutorial compares
│   ├── individual.py           # design 1: one agent decides alone
│   ├── reconcile.py            # design 2: a group discusses, then votes
│   └── tool_use.py             # design 3: one agent that calls a tool
├── utils/
│   ├── tools.py                # count_moves(): the Python function design 3 calls
│   └── log.py                  # log_prompt(), log_answer(): two text files per player per round
├── experiment/
│   └── prisoners_dilemma.py    # payoffs, prompts, the loop over rounds
├── logs/                       # one folder per run, one subfolder per round (git-ignored)
├── main.py                     # builds the two players and runs the game
└── pyproject.toml  uv.lock     # the Python environment (uv)
```

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

| Your machine | Client | First-run download | One game (5 rounds, about 40 model calls) |
|---|---|---|---|
| Apple Silicon Mac | `clients/client_mlx.py` (MLX, 4-bit model) | about 1 GB | 16 s on an M3 Ultra |
| Anything else | `clients/client.py` (PyTorch) | about 3.5 GB | 4 min on the same Mac's CPU; faster with an NVIDIA GPU |

The **first run downloads the model**, so run it once before the session, on a good connection.
A laptop without a GPU is slower than the numbers above: if a game takes too long, lower
`N_ROUNDS` in `main.py`.

You see every reply as it is produced: Alice's tool call and her move, each group member's reply in
the two discussion rounds, the group's vote, and the points.

```
=============== Round 3 ===============

[Alice] TOOL: count_moves(COOPERATE) -> In the 2 rounds played so far, the other player chose COOPERATE 1 times.

[Alice] Answer: COOPERATE
Reasoning: ...

[Bob_1, discussion round 1] Answer: COOPERATE
Reasoning: ...
...
[Bob_3, discussion round 2] Answer: COOPERATE
Reasoning: ...

[Bob] votes: {'COOPERATE': 3}

[Bob] Answer: COOPERATE

-> Alice chose COOPERATE: 3 points (total 6)
-> Bob chose COOPERATE: 3 points (total 10)
```

Every round also leaves two text files per player in `logs/<date>_<time>/round_<n>/` (git-ignored):

```
round_3/
├── Alice_prompt.txt    # what Alice read: the conversation so far, then the round's prompt
├── Alice_answer.txt    # Alice's two model calls (exact prompt and reply) and the tool's result
├── Bob_prompt.txt      # the one prompt the game sent to the group
└── Bob_answer.txt      # every member's reply in both discussion rounds, then the vote
```

**A note on the model.** This model normally writes a long hidden reasoning before every reply,
which takes minutes per reply on a laptop. To keep the tutorial fast, the client skips that
reasoning and makes the model state its answer first, then explain it. The model is tiny, so
expect its explanations to be shaky: the point here is the *mechanics* of agents and groups.

**A note on the tool call.** The model never runs anything. The agent first asks it whether it
wants the tool, and it replies with a call such as `TOOL: count_moves(DEFECT)` or with
`TOOL: none`. Python runs the call and the result comes back above the next prompt; after
`TOOL: none` the model gets the game's prompt alone. To get that reply from a model this small,
the client writes `TOOL:` as the first word and the model continues the line. Expect it to ask
for the tool nearly every time, and for the same count: judgement about when a tool helps needs
a larger model, or an API with tool calling built in.

## Things to try

All in `main.py` unless noted:

- **Swap the designs** — the three are interchangeable: `IndividualAgent`, `ReconcileAgent` and
  `ToolAgent` all take a name and a client, so any of them can play either side.
- **Change the group** — `n_members=5`, or `n_rounds=1` (no discussion: a plain majority vote).
- **Play longer** — raise `N_ROUNDS`.
- **Change the game** — edit `PAYOFFS` and `INSTRUCTIONS` in `experiment/prisoners_dilemma.py`
  (keep them in line).
- **Change the randomness** — `Client(temperature=1.0)`.
- **Write another tool** — add a function to `utils/tools.py` and call it from
  `agents/tool_use.py`; the argument the model picks comes from the game's own `options`.
