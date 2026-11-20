"""
The MLX client: the same job as client.py, for Apple Silicon Macs.

MLX is Apple's machine-learning library. It runs a 4-bit copy of the same model, which is
about twice as fast as PyTorch on a Mac and a 1 GB download instead of 3.5 GB.
"""

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from core.conversation import Conversation

MODEL_NAME = "mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit"

# We write the first characters of the model's reply ourselves (explained in client.py).
SKIP_REASONING = "</think>\n\n"
REPLY_START = "Answer:"


class Client:
    """Turns a conversation plus a new prompt into the model's reply."""

    def __init__(self, *, model_name: str = MODEL_NAME, temperature: float = 0.6, max_new_tokens: int = 150):
        self.sampler = make_sampler(temp=temperature)
        self.max_new_tokens = max_new_tokens
        self.model, self.tokenizer = load(model_name)

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
        reply = generate(
            self.model,
            self.tokenizer,
            prompt=text + SKIP_REASONING + reply_start,
            sampler=self.sampler,
            max_tokens=self.max_new_tokens,
        )
        return f"{reply_start} {reply.strip()}"
