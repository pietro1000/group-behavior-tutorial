"""
The parser: reads the answer out of a model reply.

Every reply starts with a line "Answer: <choice>".
"""

import re
from typing import List, Optional


def parse_answer(text: str, options: List[str]) -> Optional[str]:
    """The first of `options` named on the first line of the reply, e.g. "DEFECT". None if there is none."""
    first_line = text.upper().split("\n")[0]
    mentions = re.findall("|".join(options), first_line)
    return mentions[0] if mentions else None
