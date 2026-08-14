"""Integration tests: index a real corpus, then search it."""

from pathlib import Path

import pytest

from src.indexer import Indexer
from src.retriever import Retriever


@pytest.fixture
def indexed_corpus(tmp_path: Path) -> Path:
    """Build a small realistic corpus and index it. Returns processed_dir."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    (raw_dir / "math_utils.py").write_text(
        '"""Utility math functions."""\n'
        "\n"
        "import math\n"
        "\n"
        "\n"
        "def add(a: int, b: int) -> int:\n"
        '    """Return the sum of a and b."""\n'
        "    return a + b\n"
        "\n"
        "\n"
        "def circle_area(radius: float) -> float:\n"
        '    """Return the area of a circle with the given radius."""\n'
        "    return math.pi * radius ** 2\n"
        "\n"
        "\n"
        "class Vector:\n"
        '    """A simple 2D vector."""\n'
        "\n"
        "    def __init__(self, x: float, y: float) -> None:\n"
        "        self.x = x\n"
        "        self.y = y\n"
        "\n"
        "    def norm(self) -> float:\n"
        '        """Return the Euclidean norm of this vector."""\n'
        "        return math.sqrt(self.x ** 2 + self.y ** 2)\n",
        encoding="utf-8",
    )

    (raw_dir / "guide.md").write_text(
        "# Getting started\n"
        "\n"
        "This package provides simple math utilities.\n"
        "\n"
        "## Installation\n"
        "\n"
        "Run `pip install pkg` to install it.\n"
        "\n"
        "## Usage\n"
        "\n"
        "Call `add(a, b)` to add two numbers, or use the `Vector` class "
        "for 2D vectors and their norm.\n",
        encoding="utf-8",
    )

    processed_dir = tmp_path / "processed"
    Indexer().build_index(raw_dir, processed_dir, max_chunk_size=500)
    return processed_dir


def test_search_finds_the_right_python_function(indexed_corpus: Path) -> None:
    retriever = Retriever(indexed_corpus)
    results = retriever.search("how do I compute the area of a circle", k=3)

    assert len(results) > 0
    top_hit = results[0]
    assert top_hit.file_path.endswith("math_utils.py")

    # Confirm the retrieved span actually contains the right function.
    original = (indexed_corpus.parent / top_hit.file_path).read_text(encoding="utf-8")
    snippet = original[top_hit.first_character_index : top_hit.last_character_index]
    assert "circle_area" in snippet


def test_search_finds_the_right_markdown_section(indexed_corpus: Path) -> None:
    retriever = Retriever(indexed_corpus)
    results = retriever.search("how do I install this package", k=3)

    assert len(results) > 0
    assert any(r.file_path.endswith("guide.md") for r in results)


def test_search_respects_k(indexed_corpus: Path) -> None:
    retriever = Retriever(indexed_corpus)
    results = retriever.search("vector norm circle add install", k=2)
    assert len(results) <= 2


def test_search_empty_query_returns_empty(indexed_corpus: Path) -> None:
    retriever = Retriever(indexed_corpus)
    assert retriever.search("", k=5) == []
    assert retriever.search("   ", k=5) == []


def test_search_k_zero_returns_empty(indexed_corpus: Path) -> None:
    retriever = Retriever(indexed_corpus)
    assert retriever.search("circle area", k=0) == []


def test_search_punctuation_only_query_returns_empty(indexed_corpus: Path) -> None:
    retriever = Retriever(indexed_corpus)
    assert retriever.search("???", k=5) == []


def test_retriever_raises_when_no_index_built(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Retriever(tmp_path / "processed")


def test_search_with_content_returns_readable_text(indexed_corpus: Path) -> None:
    retriever = Retriever(indexed_corpus)
    chunks = retriever.search_with_content("euclidean norm vector", k=3)

    assert len(chunks) > 0
    assert any("norm" in c.content for c in chunks)