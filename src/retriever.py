from pathlib import Path
from typing import List, Optional
import bm25s
from .indexer import BM25_DIRNAME, load_chunks
from .models import Chunk, MinimalSource
from .tokenizer import tokenize


class Retriever:
    def __init__(self, processed_dir: Path) -> None:
        self.chunks: List[Chunk] = load_chunks(processed_dir)
        self.bm25: Optional[bm25s.BM25] = None
        if self.chunks:
            self.bm25 = self.load_bm25(processed_dir)

    def load_bm25(self, processed_dir: Path) -> bm25s.BM25:
        bm25_dir = processed_dir / BM25_DIRNAME
        if bm25_dir.is_dir():
            try:
                return bm25s.BM25.load(str(bm25_dir), show_progress=False)
            except (OSError, ValueError, KeyError):
                pass
        bm25 = bm25s.BM25()
        bm25.index([tokenize(c.content) for c in self.chunks],
                   show_progress=False)
        return bm25

    def search_chunks(self, query: str, k: int) -> List[Chunk]:
        if not query or not query.strip() or k <= 0 or self.bm25 is None:
            return []

        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []

        max_k = min(k, len(self.chunks))
        results, scores = self.bm25.retrieve([tokenized_query], k=max_k)
        indexes = results[0]
        return [self.chunks[i] for i in indexes]

    def search(self, query: str, k: int) -> List[MinimalSource]:
        return [c.get_minimal_source() for c in self.search_chunks(query, k)]

    def search_with_content(self, query: str, k: int) -> List[Chunk]:
        return self.search_chunks(query, k)
