"""
Agent design 3 — the tool user: one agent that can call a Python function before it decides.

A decision takes three steps:
1. First model call: we ask the model if it wants to use the tool. It replies with a tool call,
   e.g. "TOOL: count_moves(DEFECT)", or with "TOOL: none".
2. If it wrote a call, Python runs the tool (utils/tools.py) and gets a result,
   e.g. "... chose DEFECT 2 times.". If it wrote "TOOL: none", nothing runs.
3. Second model call: the model answers the game's prompt, with the tool's result above it if
   the tool ran.

The model never runs anything. It writes the name of a function, and our code runs it.
"""

from typing import List, Optional, Tuple

from core.agent import Agent
from core.conversation import Conversation
from core.parser import parse_answer
from utils.tools import count_moves

# The prompt of step 1: the game's prompt, followed by the question about the tool.
# Every reply the model can give is spelled out, so each line it can copy is a line we can read:
# given a placeholder such as count_moves(<COOPERATE or DEFECT>) it copies the placeholder.
TOOL_PROMPT = """{prompt}

Before you answer, you can use a tool that counts how many times the other player chose a move.
Do you want to use the tool? Reply with exactly one of these lines:
{calls}
TOOL: none"""


class ToolAgent(Agent):
    """One agent that can call a Python function, then decides: two model calls per decision."""

    def run(self, *, conversation: Conversation, prompt: str, options: List[str]) -> Tuple[str, str]:
        """Ask the model if it wants the tool, run the tool if it does, then ask for its answer.

        The reply is the answer of step 3. The transcript holds the three steps, with the exact
        prompt and the exact reply of both model calls.
        """
        # Step 1: first model call. The model writes a tool call, or "TOOL: none".
        # One possible call per option: "TOOL: count_moves(COOPERATE)", "TOOL: count_moves(DEFECT)".
        calls = "\n".join(f"TOOL: count_moves({option})" for option in options)
        tool_prompt = TOOL_PROMPT.format(prompt=prompt, calls=calls)
        # reply_start="TOOL:" makes the client write "TOOL:" as the first word of the reply, so the
        # model continues with a call or with "none". With the default start, "Answer:", this small
        # model names a move instead (reply_start is explained in clients/client.py).
        tool_reply = self.client.generate(conversation=conversation, prompt=tool_prompt, reply_start="TOOL:")

        # Step 2: Python runs the tool, if the model asked for it.
        action = self.read_tool_call(tool_reply, options)
        if action is None:
            ran = "no tool runs"
            result = "The reply is not a tool call, so nothing runs."
            answer_prompt = prompt
        else:
            ran = f"Python runs count_moves({action})"
            result = count_moves(conversation=conversation, action=action)
            # The model reads the tool's result above the game's prompt.
            answer_prompt = f"{result}\n\n{prompt}"
        # The model's own line, so that what it wrote shows next to what we did with it.
        print(f"\n[{self.name}] {tool_reply.splitlines()[0]} -> {result}")

        # Step 3: second model call. The model answers the game's prompt.
        # The tool exchange stays out of the conversation: the game owns it, and it holds only moves.
        reply = self.client.generate(conversation=conversation, prompt=answer_prompt)

        # Both model calls read the conversation so far (the game logs it in <name>_prompt.txt)
        # followed by the prompt shown here.
        transcript = (
            f"[{self.name}, step 1: the prompt we send to the model]\n{tool_prompt}\n\n"
            f"[{self.name}, step 1: the model's reply]\n{tool_reply}\n\n"
            f"[{self.name}, step 2: {ran}]\n{result}\n\n"
            f"[{self.name}, step 3: the prompt we send to the model]\n{answer_prompt}\n\n"
            f"[{self.name}, step 3: the model's reply]\n{reply}"
        )
        return reply, transcript

    @staticmethod
    def read_tool_call(tool_reply: str, options: List[str]) -> Optional[str]:
        """The option inside the tool call on the first line of the reply, e.g. "DEFECT".

        None if the first line is "TOOL: none", or anything else that is not a call we can run.
        Only the first line counts: the model may keep writing after it.
        """
        first_line = tool_reply.splitlines()[0]
        if "count_moves" not in first_line:
            return None
        return parse_answer(first_line, options)
