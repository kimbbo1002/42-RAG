from pathlib import Path
from typing import Any, List, Optional
from .models import MinimalSource


SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about a codebase. "
    "Answer ONLY using the sources provided below. If the sources do not "
    "contain the answer, say so plainly instead of guessing. Be concise "
    "and directly answer the question asked."
)
MODEL_NAME = "Qwen/Qwen3-0.6B"


class Generator:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        