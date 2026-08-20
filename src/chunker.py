from typing import List
from .models import Chunk
import ast
import markdown_it


CHUNK_OVERLAP = 200


# when chunk needs to be split by the max_chunk_size
def split_by_max(
        text: str,
        max_size: int,
        start_index: int = 0,
        overlap: int = CHUNK_OVERLAP
) -> List[tuple[str, int, int]]:
    if max_size <= 0 or len(text) <= max_size:
        return [(text, start_index, start_index + len(text))]

    overlap = max(0, min(overlap, max_size // 2))
    step = max(1, max_size - overlap)
    chunk: List[tuple[str, int, int]] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + max_size, text_len)
        chunk.append((text[start:end], start_index + start, start_index + end))
        if end == text_len:
            break
        start += step
    return chunk


# ast lines are 1 indexed
# therefore lines[i] = character index where line i + 1 starts
def get_line_index(text: str) -> List[int]:
    lines = [0]
    for line in text.splitlines(keepends=True):
        lines.append(lines[-1] + len(line))
    return lines


# class defined to chunk the python files
class PythonChunker:

    # fallback method to be called when error occurs
    # chunking is done according to max_chunk_size
    def fallback(
            self, file_path: str,
            content: str,
            max_chunk_size: int
    ) -> List[Chunk]:
        return [
            Chunk(
                file_path=file_path,
                content=text,
                first_character_index=start,
                last_character_index=end,
                chunk_type="py"
            )
            for text, start, end in split_by_max(content, max_chunk_size)
        ]

    # takes part of a text and actually stores it as a chunk
    def stock_chunk(
            self, chunks: List[Chunk],
            file_path: str,
            content: str,
            start: int,
            end: int,
            max_chunk_size: int
    ) -> None:
        block = content[start:end]
        if not block.strip():
            return
        for b_text, b_start, b_end in split_by_max(block, max_chunk_size,
                                                   start):
            chunks.append(
                Chunk(
                    file_path=file_path,
                    content=b_text,
                    first_character_index=b_start,
                    last_character_index=b_end,
                    chunk_type="py")
            )

    # function that manages the chunking flow
    def get_chunk(
            self, file_path: str,
            content: str,
            max_chunk_size: int
    ) -> List[Chunk]:
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return self.fallback(file_path, content, max_chunk_size)

        line_idx = get_line_index(content)
        chunks: List[Chunk] = []

        buffer_start_idx: int | None = None
        buffer_end_idx: int | None = None

        def catch_nondef() -> None:
            nonlocal buffer_start_idx, buffer_end_idx
            if buffer_start_idx is None or buffer_end_idx is None:
                return

            start = line_idx[buffer_start_idx - 1]
            end = min(line_idx[buffer_end_idx], len(content))

            self.stock_chunk(chunks, file_path, content, start, end,
                             max_chunk_size)
            buffer_start_idx = None
            buffer_end_idx = None

        for node in tree.body:
            is_def = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                       ast.ClassDef))
            if is_def:
                catch_nondef()
                start = line_idx[node.lineno - 1]
                end_lineno = getattr(node, "end_lineno", node.lineno)
                end = min(line_idx[end_lineno], len(content))
                self.stock_chunk(chunks, file_path, content, start, end,
                                 max_chunk_size)
            else:
                if buffer_start_idx is None:
                    buffer_start_idx = node.lineno
                buffer_end_idx = getattr(node, "end_lineno", node.lineno)

        catch_nondef()

        if not chunks:
            return self.fallback(file_path, content, max_chunk_size)
        return chunks


md_parser = markdown_it.MarkdownIt()


class MarkdownChunker:

    def get_header_index(
            self, content: str,
            line_idx: List[int]
    ) -> List[int]:
        tokens = md_parser.parse(content)
        header_idx = []
        for token in tokens:
            if token.type == "heading_open" and token.map is not None:
                header_idx.append(line_idx[token.map[0]])
        return header_idx

    def stock_chunk(
            self, file_path: str,
            content: str,
            start: int,
            end: int
    ) -> Chunk:
        return Chunk(
            file_path=file_path,
            content=content,
            first_character_index=start,
            last_character_index=end,
            chunk_type="md"
        )

    def fallback(
                self, file_path: str,
                content: str,
                max_chunk_size: int
    ) -> List[Chunk]:
        return [
            Chunk(
                file_path=file_path,
                content=text,
                first_character_index=start,
                last_character_index=end,
                chunk_type="md"
            )
            for text, start, end in split_by_max(content, max_chunk_size)
        ]

    def get_chunk(
            self, file_path: str,
            content: str,
            max_chunk_size: int
    ) -> List[Chunk]:
        line_idx = get_line_index(content)
        header_idx = self.get_header_index(content, line_idx)
        chunks: List[Chunk] = []

        if not header_idx:
            return self.fallback(file_path, content, max_chunk_size)

        section_idx: List[tuple[int, int]] = []
        if header_idx[0] > 0:
            section_idx.append((0, header_idx[0]))
        for i, start in enumerate(header_idx):
            end = (header_idx[i + 1] if i + 1 < len(header_idx)
                   else len(content))
            section_idx.append((start, end))

        for s, e in section_idx:
            section_text = content[s:e]
            if not section_text.strip():
                continue
            for text, b_start, b_end in split_by_max(section_text,
                                                     max_chunk_size, s):
                if text.strip():
                    chunks.append(self.stock_chunk(file_path, text,
                                                   b_start, b_end))
        return chunks
