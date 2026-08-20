import json
from typing import List
from pathlib import Path
import bm25s
from tqdm import tqdm
from .models import Chunk
from .chunker import PythonChunker, MarkdownChunker
from .tokenizer import tokenize


PY_EXTENSION = {".py"}
MD_EXTENSION = {".md", ".markdown", ".txt"}
INDEX_FILENAME = "chunks.json"
BM25_DIRNAME = "bm25"


class Indexer:
    def __init__(self) -> None:
        self.py_chunker = PythonChunker()
        self.md_chunker = MarkdownChunker()

    def get_files(
            self, raw_dir: Path
    ) -> List[Path]:
        files: List[Path] = []
        file_extensions = PY_EXTENSION | MD_EXTENSION
        for p in raw_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in file_extensions:
                files.append(p)
        return files

    def build_index(
            self, raw_dir: Path,
            processed_dir: Path,
            max_chunk_size: int
    ) -> int:
        if not raw_dir.exists():
            raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

        files = self.get_files(raw_dir)
        all_chunks: List[Chunk] = []

        for path in tqdm(files, desc="Chunking", unit="file"):
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            file_path = path.as_posix()
            file_ext = path.suffix.lower()
            if file_ext in PY_EXTENSION:
                chunks = self.py_chunker.get_chunk(file_path, content,
                                                   max_chunk_size)
            elif file_ext in MD_EXTENSION:
                chunks = self.md_chunker.get_chunk(file_path, content,
                                                   max_chunk_size)
            else:
                continue
            all_chunks.extend(chunks)

        processed_dir.mkdir(parents=True, exist_ok=True)
        chunks_dict = [c.model_dump() for c in all_chunks]
        output_path = processed_dir / INDEX_FILENAME
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(chunks_dict, file)

        corpus_tokens = [
            tokenize(c.content)
            for c in tqdm(all_chunks, desc="Tokenizing", unit="chunk")
        ]
        if corpus_tokens:
            bm25 = bm25s.BM25()
            bm25.index(corpus_tokens, show_progress=False)
            bm25.save(str(processed_dir / BM25_DIRNAME), show_progress=False)

        return len(all_chunks)


def load_chunks(processed_dir: Path) -> List[Chunk]:
    index_path = processed_dir / INDEX_FILENAME
    if not index_path.exists():
        raise FileNotFoundError(
            f"No index found at {index_path}, run 'index' first"
        )
    with index_path.open("r", encoding="utf-8") as file:
        raw_chunks = json.load(file)

    chunks: List[Chunk] = []
    for c in raw_chunks:
        chunks.append(Chunk.model_validate(c))

    return chunks
