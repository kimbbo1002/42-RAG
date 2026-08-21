import json
from pathlib import Path
from typing import Any, Dict, List, Optional, cast
from tqdm import tqdm
from .generator import Generator
from .indexer import Indexer
from .models import (
    AnsweredQuestion,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer
)
from .retriever import Retriever


class CLI:
    def init_retriever(self, processed_dir: str) -> Optional[Retriever]:
        try:
            return Retriever(Path(processed_dir))
        except FileNotFoundError as e:
            print(f"Could not load index: {e}")
            return None

    def print_source(
            self, source: MinimalSource, indent: str = ""
    ) -> None:
        span = (
            f"[{source.first_character_index}:{source.last_character_index}]"
        )
        print(f"{indent}{source.file_path} {span}")

    def read_json(self, path: str) -> Optional[Dict[str, Any]]:
        file_path = Path(path)
        if not file_path.exists():
            print(f"File not found: {path}")
            return None

        try:
            with file_path.open("r", encoding="utf-8") as f:
                return cast(Dict[str, Any], json.load(f))
        except json.JSONDecodeError as e:
            print(f"Malformed JSON in {path}: {e}")
            return None

    def write_json(
            self, save_dir: str,
            source_path: str,
            model: Any
    ) -> Optional[Path]:
        try:
            out_dir = Path(save_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / Path(source_path).name
            with out_path.open("w", encoding="utf-8") as f:
                f.write(model.model_dump_json(indent=2))
            return out_path
        except OSError as e:
            print(f"Could not write output to {save_dir}: {e}")
            return None

    def load_dataset(self, path: str) -> Optional[RagDataset]:
        data = self.read_json(path)
        if data is None:
            return None
        try:
            return RagDataset.model_validate(data)
        except Exception as e:
            print(f"Dataset at {path} does not match expected format: {e}")
            return None

    def load_student_search_res(
            self, path: str
    ) -> Optional[StudentSearchResults]:
        data = self.read_json(path)
        if data is None:
            return None
        try:
            return StudentSearchResults.model_validate(data)
        except Exception as e:
            print(f"Student search results do not match expected format: {e}")
            return None

    def index(
            self, max_chunk_size: int = 2000,
            raw_dir: str = "data/raw",
            processed_dir: str = "data/processed"
    ) -> None:
        if max_chunk_size <= 0:
            print("max_chunk_size must be a positive integer.")
            return
        try:
            indexer = Indexer()
            indexer.build_index(
                Path(raw_dir), Path(processed_dir), max_chunk_size
            )
            print(f"Ingestion complete! Indices saved under {processed_dir}/")
        except FileNotFoundError as e:
            print(f"Indexing failed: {e}")
        except Exception as e:
            print(f"Indexing failed unexpectedly: {e}")

    def search(
            self, query: str,
            k: int = 5,
            processed_dir: str = "data/processed"
    ) -> None:
        retriever = self.init_retriever(processed_dir)
        if retriever is None:
            return

        results = retriever.search(query, k)
        if not results:
            print("No results found (empty query, k<=0, or empty index)")
            return
        for s in results:
            self.print_source(s)

    def answer(
            self, query: str,
            k: int = 5,
            processed_dir: str = "data/processed",
            raw_dir: str = "data/raw"
    ) -> None:
        retriever = self.init_retriever(processed_dir)
        if retriever is None:
            return

        sources = retriever.search(query, k)
        if not sources:
            print("No sources retrieved, cannot generate an answer.")
            return

        generator = Generator()
        answer_text = generator.answer(query, sources, Path(raw_dir))
        print("Sources:")
        for s in sources:
            self.print_source(s, "   ")
        print(f"\nAnswer:\n{answer_text}")

    def search_dataset(
            self, dataset_path: str,
            k: int = 10,
            save_directory: str = "data/output/search_results",
            processed_dir: str = "data/processed"
    ) -> None:
        dataset = self.load_dataset(dataset_path)
        if dataset is None:
            return
        retriever = self.init_retriever(processed_dir)
        if retriever is None:
            return

        results: List[MinimalSearchResults] = []
        for question in tqdm(dataset.rag_questions, desc="Searching",
                             unit="question"):
            sources = retriever.search(question.question, k)
            results.append(
                MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=sources
                )
            )

        output = StudentSearchResults(search_results=results, k=k)
        output_path = self.write_json(save_directory, dataset_path, output)
        if output_path:
            print(f"Saved student_search_results to {output_path}")

    def answer_dataset(
            self, student_search_results_path: str,
            save_directory: str = "data/output/search_results_and_answer",
            raw_dir: str = "data/raw"
    ) -> None:
        student_res = self.load_student_search_res(student_search_results_path)
        if student_res is None:
            return

        generator = Generator()
        total_count = len(student_res.search_results)
        answers: List[MinimalAnswer] = []
        for a in tqdm(student_res.search_results, desc="Answering",
                      unit="question"):
            answer_text = generator.answer(
                a.question, a.retrieved_sources, Path(raw_dir)
            )
            answers.append(
                MinimalAnswer(
                    question_id=a.question_id,
                    question=a.question,
                    retrieved_sources=a.retrieved_sources,
                    answer=answer_text
                )
            )

        output = StudentSearchResultsAndAnswer(
            search_results=answers,
            k=student_res.k
        )
        output_path = self.write_json(
            save_directory, student_search_results_path, output
        )
        if output_path:
            print(
                f"Loaded {total_count} questions ... "
                f"Processed {total_count} of {total_count} questions"
            )
            print(f"Saved student_search_results_and_answer to {output_path}")

    def evaluate(
            self, student_search_results_path: str,
            dataset_path: str
    ) -> None:
        student_res = self.load_student_search_res(student_search_results_path)
        real_res = self.load_dataset(dataset_path)

        if student_res is None or real_res is None:
            print("Evaluation aborted due to loading errors.")
            return

        real_dict = {q.question_id: q for q in real_res.rag_questions}
        question_count = len(student_res.search_results)

        if question_count == 0:
            print("No search result to evaluate.")
            return

        recall_count = 0
        for s in student_res.search_results:
            real_q = real_dict.get(s.question_id)
            if not real_q or not isinstance(real_q, AnsweredQuestion):
                continue

            found = False
            for retrieved in s.retrieved_sources[:student_res.k]:
                for target in real_q.sources:
                    if retrieved.file_path != target.file_path:
                        continue

                    inter_start = max(
                        retrieved.first_character_index,
                        target.first_character_index
                    )
                    inter_end = min(
                        retrieved.last_character_index,
                        target.last_character_index
                    )
                    intersection = max(0, inter_end - inter_start)

                    if intersection == 0:
                        continue

                    union = (
                        (
                            retrieved.last_character_index
                            - retrieved.first_character_index
                        )
                        + (
                            target.last_character_index
                            - target.first_character_index
                        )
                        - intersection
                    )
                    iou = intersection / union if union > 0 else 0.0

                    if iou > 0.05:
                        found = True
                        break
                if found:
                    break
            if found:
                recall_count += 1
        recall_at_k = recall_count / question_count
        print("Local Evaluation Results")
        print("=" * 40)
        print(f"Total questions: {question_count}")
        print(f"Recall@{student_res.k}: {recall_at_k:.3f}")
