"""Tests for src.indexer."""

from pathlib import Path

import pytest

from src.indexer import Indexer, load_chunks
from src.models import Chunk


@pytest.fixture
def raw_corpus(tmp_path: Path) -> Path:
    """Build a tiny raw corpus with one Python file and one Markdown file."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    (raw_dir / "math_utils.py").write_text(
        '"""Utility math functions."""\n'
        "\n"
        "def add(a: int, b: int) -> int:\n"
        '    """Return the sum of a and b."""\n'
        "    return a + b\n",
        encoding="utf-8",
    )

    (raw_dir / "guide.md").write_text(
        "# Getting started\n"
        "\n"
        "This package provides simple math utilities.\n"
        "\n"
        "## Installation\n"
        "\n"
        "Run `pip install pkg` to install it.\n",
        encoding="utf-8",
    )

    # A file type that should be skipped entirely.
    (raw_dir / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    return raw_dir


def test_build_index_creates_chunks_json(tmp_path: Path, raw_corpus: Path) -> None:
    processed_dir = tmp_path / "processed"
    indexer = Indexer()

    count = indexer.build_index(raw_corpus, processed_dir, max_chunk_size=500)

    assert count > 0
    assert (processed_dir / "chunks.json").exists()


def test_build_index_skips_unsupported_extensions(tmp_path: Path, raw_corpus: Path) -> None:
    processed_dir = tmp_path / "processed"
    indexer = Indexer()
    indexer.build_index(raw_corpus, processed_dir, max_chunk_size=500)

    chunks = load_chunks(processed_dir)
    assert all(not c.file_path.endswith(".png") for c in chunks)


def test_build_index_raises_on_missing_raw_dir(tmp_path: Path) -> None:
    indexer = Indexer()
    with pytest.raises(FileNotFoundError):
        indexer.build_index(tmp_path / "does_not_exist", tmp_path / "processed", 500)


def test_load_chunks_raises_when_no_index_built(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_chunks(tmp_path / "processed")


def test_load_chunks_round_trips_offsets(tmp_path: Path, raw_corpus: Path) -> None:
    """The most important invariant: content[start:end] must equal chunk.content."""
    processed_dir = tmp_path / "processed"
    Indexer().build_index(raw_corpus, processed_dir, max_chunk_size=500)

    chunks = load_chunks(processed_dir)
    assert len(chunks) > 0

    for chunk in chunks:
        assert isinstance(chunk, Chunk)
        original_text = Path(chunk.file_path).read_text(encoding="utf-8")
        sliced = original_text[chunk.first_character_index : chunk.last_character_index]
        assert sliced == chunk.content


def test_no_chunk_exceeds_max_chunk_size(tmp_path: Path, raw_corpus: Path) -> None:
    processed_dir = tmp_path / "processed"
    max_size = 60  # deliberately small, to force sliding-window splitting
    Indexer().build_index(raw_corpus, processed_dir, max_size)

    chunks = load_chunks(processed_dir)
    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk.content) <= max_size


def test_chunk_kinds_are_correct(tmp_path: Path, raw_corpus: Path) -> None:
    processed_dir = tmp_path / "processed"
    Indexer().build_index(raw_corpus, processed_dir, max_chunk_size=500)

    chunks = load_chunks(processed_dir)
    py_chunks = [c for c in chunks if c.file_path.endswith(".py")]
    md_chunks = [c for c in chunks if c.file_path.endswith(".md")]

    assert py_chunks and all(c.chunk_type == "py" for c in py_chunks)
    assert md_chunks and all(c.chunk_type == "md" for c in md_chunks)