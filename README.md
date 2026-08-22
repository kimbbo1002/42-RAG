*This project has been created as part of the 42 curriculum by bokim.*

# RAG Against the Machine

## Description
This project builds a pipeline that ingests and indexes a codebase, retrieves the most relevant snippets for a given question, passes them to a small local model to generate a grounded answer, and measures retrieval quality using recall@k.

## Instructions
### Makefile rules
- `make install` — installs all dependencies with uv sync.
- `make run` — runs an example search query through the CLI.
- `make debug` — runs that same search query, but inside pdb so it can be stepped through.
- `make clean` — deletes cache files (__pycache__, .mypy_cache, *.pyc).
- `make fclean` — runs clean, then also removes uv.lock and the .venv folder, for a full reset.
- `make lint` — checks code style with flake8 and type-checks it with mypy.

### Example usage
#### Indexing
```bash
uv run python -m src index --max_chunk_size 2000
```

### Searching dataset
```bash
uv run python -m src search_dataset
--dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json
--k 10
--save_directory data/output/search_results/UnansweredQuestions
```

### Scoring with moulinette
```bash
./moulinette evaluate_student_search_results
data/output/search_results/UnansweredQuestions/dataset_docs_public.json
data/datasets/AnsweredQuestions/dataset_docs_public.json
--k 10 --max_context_length 2000
```

### Generating answers
```bash
uv run python -m src answer_dataset
--student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json
--save_directory data/output/search_results_and_answer/UnansweredQuestions
```

## System architecture
The system is organized into six modules under `src/`, driven by a Python Fire CLI, and operates on two distinct timelines: indexing runs once and writes its results to disk, while querying runs once per question and reads what indexing left behind.

```
                 INDEX TIME (once)              QUERY TIME (per question)

                 data/raw/                      question
                     │                              │
                     ▼                              ▼
        PythonChunker / MarkdownChunker         tokenize()  ◄── same function
                     │                              │
                     ▼                              ▼
              19 048 Chunk objects            BM25.retrieve(k)  ◄── loads bm25/
                     │                              │
                     ▼                              ▼
          data/processed/chunks.json          k × MinimalSource
                     │                              │
                     ▼                              ▼
                tokenize()                     build_context()  ──► re-reads data/raw/
                     │                              │
                     ▼                              ▼
          data/processed/bm25/               Qwen3-0.6B (chat template)
                                                    │
                                                    ▼
                                               answer string
```

| Module | Responsibility |
| --- | --- |
| `src/models.py` | All pydantic models: `MinimalSource`, `RagDataset`, `StudentSearchResults`, `StudentSearchResultsAndAnswer`, and the internal `Chunk` |
| `src/chunker.py` | The two chunking strategies and the shared `split_by_max` size enforcement |
| `src/tokenizer.py` | `tokenize()` and the stopword list — shared by both timelines |
| `src/indexer.py` | Walks the corpus, chunks it, persists `chunks.json` and the BM25 index |
| `src/retriever.py` | Loads the persisted index and answers top-k queries |
| `src/generator.py` | Rebuilds source text from spans and prompts Qwen3-0.6B |
| `src/cli.py` | The six commands, input validation, and JSON output |

## Chunking strategy
A Python file and a Markdown page don't break apart the same way, so each is handled by its own strategy. Both fall back to fixed-size splitting whenever their expected structure is absent.

**Python — one chunk per top-level definition.** `ast.parse` gives the line span of every top-level node. Functions and classes each become a chunk; consecutive non-definition nodes (imports, constants, the module docstring) accumulate into a buffer flushed as a single chunk when the next definition appears. A `SyntaxError` falls back to fixed-size splitting.

**Markdown / text — one chunk per heading section.** `markdown-it-py` reports the line of every `heading_open` token. Each heading starts a section running to the next heading; any preamble before the first heading becomes its own chunk. A page with no headings falls back to fixed-size splitting. `.md`, `.markdown` and `.txt` all use this path.

**Size enforcement.** Every chunk then passes through `split_by_max`, which enforces
`--max_chunk_size` with a 200-character overlap so a sentence cut at a boundary still appears intact in one of the two pieces. The overlap is clamped to half the chunk size — without that clamp, a `--max_chunk_size` of 200 or less makes the step size collapse to 1 and produces roughly one chunk per character.

## Retrieval method
Retrieval uses **BM25** via `bm25s`, with the index persisted under `data/processed/bm25/` so that a query loads it instead of rebuilding it from scratch.

Tokenization is kept deliberately simple: `[A-Za-z0-9_]+`, lowercased, with a hand-picked English stopword list removed. Keeping the underscore inside the token class lets identifiers such as `use_fast` or `tie_word_embeddings` survive as single tokens, so they can be matched verbatim when a question quotes them directly.

Of all the design choices here, the stopword list carries the most weight — and for a reason that isn't immediately obvious. BM25 already down-weights common terms through IDF, but IDF measures rarity in the corpus, not usefulness to the query.

Nearly every question contains "what" or "how," while source files almost never do. As a result, BM25 was rating the question word as more informative than the actual subject of the question, and the handful of chunks that happened to contain prose like "What is vLLM?" picked up a large, spurious boost on every single query. Removing stopwords moved 14 questions into the top five results.

A query made up entirely of stopwords tokenizes to an empty list and returns no results. This is intentional: a query with no content words has nothing left to match against, and returning boilerplate results would be worse than returning none at all.


## Performance analysis

### Recall@k, as reported by the provided moulinette

| Dataset | Recall@1 | Recall@3 | **Recall@5** | Recall@10 | Threshold |
| --- | --- | --- | --- | --- | --- |
| docs (100 questions) | 0.650 | 0.790 | **0.840** | 0.900 | ≥ 0.80 ✅ |
| code (99 questions) | 0.313 | 0.505 | **0.556** | 0.626 | ≥ 0.50 ✅ |

## Design decisions

**BM25 over TF-IDF.** BM25 is an advanced, non-linear evolution of TF-IDF that fixes core flaws regarding word repetition and document length.

**Persisting the BM25 index, not just the chunks.** The index command writes both `chunks.json `and a serialized BM25 structure to disk. Loading the index instead of rebuilding it cut retriever startup time from 0.9 s down to 0.11 s. If `bm25/` is missing or unreadable, the retriever simply rebuilds it in memory rather than failing outright, so an index directory from an older run still works.

**Lazy model loading.** Neither `search` nor `search_dataset` construct the transformers pipeline, so retrieval-only work never pays for a multi-second model load — or a 1.2 GB download on a cold machine.

## Challenges faced

**An overlap constant larger than the chunk size.** `split_by_max` originally used a fixed 200-character overlap. Any `--max_chunk_size` at or below 200 made the step size collapse to 1, producing roughly one chunk per character — 9,901 chunks from just 10,000 characters, and an out-of-memory condition on the real corpus. The fix was to clamp the overlap to half the chunk size.

## Resources
- https://www.ai-bites.net/tf-idf-and-bm25-for-rag-a-complete-guide/

### How AI was used
- understanding modules used such as `ast` and understanding new concepts such as chunking and indexing
