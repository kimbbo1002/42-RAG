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


def build_context(
        sources: List[MinimalSource],
        raw_dir: Path
) -> str:
    source_texts = []
    for s in sources:
        possible_paths = [Path(s.file_path), raw_dir / s.file_path]
        content = None
        for p in possible_paths:
            if p.exists():
                try:
                    content = p.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    content = None
                break
        if content is None:
            continue
        text = content[s.first_character_index:s.last_character_index]
        source_texts.append(f"### Source: {s.file_path}\n{text}")
    return "\n\n".join(source_texts)


class Generator:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self.pipeline = Optional[Any] = None

    def is_loaded(self) -> None:
        if self.pipeline is not None:
            return
        from transformers import pipeline
        self.pipeline = pipeline(model=MODEL_NAME)

    def answer(
            self, question: str,
            sources: List[MinimalSource],
            raw_dir: Path
    ) -> str:
        context = build_context(sources, raw_dir)
        if not context.strip():
            return "Could not find relevant source content."

        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"--- Sources ---\n{context}\n --- End sources ---\n\n"
            f"Question: {question}\nAnswer:"
        )
        try:
            self.is_loaded()
            output = self.pipeline(
                prompt,
                max_new_tokens=256,
            )[0]["generated_text"]
            return output
        except Exception as e:
            return f"Answer generation failed: {e}"
            