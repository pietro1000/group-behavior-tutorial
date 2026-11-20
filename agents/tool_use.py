"""
Agent design 3 — the tool user: one agent that calls a Python function before it decides.

A decision takes three steps:
1. First model call: the model writes a tool call, e.g. "TOOL: count_moves(DEFECT)".
2. Python runs the tool (utils/tools.py) and gets a result, e.g. "... chose DEFECT 2 times."
3. Second model call: the model reads the result and answers the game's prompt.

The model never runs anything. It writes the name of a function, and our code runs it.
"""

from typing import List, Tuple

from core.agent import Agent
from core.conversation import Conversation
from core.parser import parse_answer
from utils.tools import count_moves

# The prompt of step 1: the game's prompt, followed by the tool calls the model can write.
# Every call is spelled out, so each line the model can copy is a call we can read: given a
# placeholder such as count_moves(<COOPERATE or DEFECT>) it copies the placeholder.
TOOL_PROMPT = """{prompt}

Before you answer, you may call a tool that counts the other player's past moves:
{calls}"""


class ToolAgent(Agent):
    """One agent that calls a Python function, then decides: two model calls per decision."""

    def run(self, *, conversation: Conversation, prompt: str, options: List[str]) -> Tuple[str, str]:
        """Ask the model for a tool call, run the tool, then ask the model for its answer.

        The reply is the answer of step 3. The transcript holds all three steps.
        """
        # Step 1: first model call. The model writes a tool call.
        # One possible call per option: "TOOL: count_moves(COOPERATE)", "TOOL: count_moves(DEFECT)".
        calls = "\n".join(f"TOOL: count_moves({option})" for option in options)
        tool_prompt = TOOL_PROMPT.format(prompt=prompt, calls=calls)
        # reply_start="TOOL:" makes the client write "TOOL:" as the first word of the reply, so the
        # model continues with a call. With the default start, "Answer:", this small model names a
        # move even when it was asked for a tool (reply_start is explained in clients/client.py).
        tool_reply = self.client.generate(conversation=conversation, prompt=tool_prompt, reply_start="TOOL:")
        # The model does not stop after the call: it goes on to answer the prompt, before any tool
        # has run. Only the first line is the tool call, so we keep that line and drop the rest.
        tool_call = tool_reply.splitlines()[0]

        # Step 2: Python runs the tool the model asked for.
        # parse_answer finds the option inside "TOOL: count_moves(DEFECT)". If the call names no
        # option we cannot read it, and we count the first option instead.
        action = parse_answer(tool_call, options) or options[0]
        result = count_moves(conversation=conversation, action=action)
        # The model's own line, so an unreadable call shows up instead of being papered over.
        print(f"\n[{self.name}] {tool_call} -> {result}")

        # Step 3: second model call. The model decides, with the tool's result above the prompt.
        # The tool exchange stays out of the conversation: the game owns it, and it holds only moves.
        reply = self.client.generate(conversation=conversation, prompt=f"{result}\n\n{prompt}")

        transcript = (
            f"[{self.name}, step 1: the model writes a tool call]\n{tool_call}\n\n"
            f"[{self.name}, step 2: Python runs count_moves({action})]\n{result}\n\n"
            f"[{self.name}, step 3: the model answers]\n{reply}"
        )
        return reply, transcript
