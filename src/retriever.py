import re
from pathlib import Path
from typing import List
from rank_bm25 import BM25Okapi
from .indexer import load_chunks
from .models import Chunk, MinimalSource


TOKEN_REGEX = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    tokens = []
    for t in TOKEN_REGEX.findall(text):
        tokens.append(t.lower())
    return tokens


class Retriever:
    def __init__(self, processed_dir: Path) -> None:
        self.chunks: List[Chunk] = load_chunks(processed_dir)
        tokens = [tokenize(c.content) for c in self.chunks]
        self.bm25 = BM25Okapi(tokens) if tokens else None

    def search(self, query: str, k: int) -> List[MinimalSource]:
        if not query or not query.strip() or k <= 0 or self.bm25 is None:
            return []

        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]
        return [self.chunks[r].get_minimal_source() for r in ranked]

    def search_with_content(self, query: str, k: int) -> List[Chunk]:
        if not query or not query.strip() or k <= 0 or self.bm25 is None:
            return []

        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []

        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]
        return [self.chunks[r] for r in ranked]