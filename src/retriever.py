import re
from pathlib import Path
from typing import List, Optional
import bm25s
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
        self.bm25: Optional[bm25s.BM25] = None
        if self.chunks:
            tokens = [tokenize(c.content) for c in self.chunks]
            self.bm25 = bm25s.BM25()
            self.bm25.index(tokens)

    def search_chunks(self, query: str, k: int) -> List[Chunk]:
        if not query or not query.strip() or k <= 0 or self.bm25 is None:
            return []

        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []

        max_k = min(k, len(self.chunks))
        results, scores = self.bm25.retrieve([tokenized_query], max_k)
        indexes = results[0]
        return [self.chunks[i] for i in indexes]

    def search(self, query: str, k: int) -> List[MinimalSource]:
        return [c.get_minimal_source() for c in self.search_chunks(query, k)]

    def search_with_content(self, query: str, k: int) -> List[Chunk]:
        return self.search_chunks(query, k)
