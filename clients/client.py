"""
The client: the code that talks to the language model.

It loads a small open model from Hugging Face and runs it on this machine with PyTorch: no API
key, no cloud. It works on any machine; client_mlx.py does the same job faster on a Mac.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.conversation import Conversation

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# We write the first characters of the model's reply ourselves:
# - "</think>" closes the reasoning block, which this model would otherwise fill with
#   thousands of tokens before replying (minutes per reply on a laptop);
# - "Answer:" makes it state its choice first, in the format core/parser.py reads.
# An agent can ask for a different start: agents/tool_use.py passes "TOOL:" to get a tool call.
SKIP_REASONING = "</think>\n\n"
REPLY_START = "Answer:"


class Client:
    """Turns a conversation plus a new prompt into the model's reply."""

    def __init__(self, *, model_name: str = MODEL_NAME, temperature: float = 1, max_new_tokens: int = 600):
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        # Use the GPU if this machine has one (Apple or NVIDIA), otherwise the CPU.
        self.device = torch.accelerator.current_accelerator() or "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

    def generate(self, *, conversation: Conversation, prompt: str, reply_start: str = REPLY_START) -> str:
        """Send the conversation so far plus `prompt` as a new user message; return the reply.

        reply_start is the beginning of the reply, and we write it instead of the model. The model
        reads it as its own words and continues from there, so every reply starts with reply_start.
        The text the model reads ends like this:

            ...the prompt... </think>

            Answer:

        - "Answer:" (the default): the model continues with its choice, e.g. " DEFECT".
        - "TOOL:" (passed by agents/tool_use.py): the model continues with a tool call,
          e.g. " count_moves(DEFECT)".
        """
        messages = conversation.messages + [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.tokenizer(
            text + SKIP_REASONING + reply_start, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)
        output = self.model.generate(
            **inputs, do_sample=True, temperature=self.temperature, max_new_tokens=self.max_new_tokens
        )
        # The output holds the prompt followed by the new tokens: decode only the new ones.
        reply = self.tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return f"{reply_start} {reply.strip()}"
