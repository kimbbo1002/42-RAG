from src.chunker import PythonChunker, MarkdownChunker

py_file = "chunk_test.py"
md_file = "chunk_test.md"

py_content = open(py_file).read()
py_chunks = PythonChunker().get_chunk(py_file, py_content, 500)

md_content = open(md_file).read()
md_chunks = MarkdownChunker().get_chunk(md_file, md_content, 500)

print("=== TESTING PythonChunker ===")
for i, c in enumerate(py_chunks):
    print(f"{i}: ", c.chunk_type, c.first_character_index, c.last_character_index, repr(c.content[:50]))

print("=== TESTING MarkdownChunker ===")
for i, c in enumerate(md_chunks):
    print(f"{i}: ", c.chunk_type, c.first_character_index, c.last_character_index, repr(c.content[:50]))
