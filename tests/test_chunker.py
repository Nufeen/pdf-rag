import pytest

from pdf_rag.chunker import split_text


def test_empty_text_returns_empty_list():
    assert split_text("", chunk_size=100, overlap=10) == []


def test_text_shorter_than_chunk_size_returned_as_single_chunk():
    result = split_text("Hello world.", chunk_size=100, overlap=10)
    assert result == ["Hello world."]


def test_long_text_produces_multiple_chunks():
    text = "word " * 200  # ~1000 chars
    result = split_text(text, chunk_size=100, overlap=10)
    assert len(result) > 1


def test_all_chunks_within_size_with_overlap_tolerance():
    text = "word " * 200
    chunk_size = 100
    overlap = 10
    result = split_text(text, chunk_size=chunk_size, overlap=overlap)
    for chunk in result:
        # chunks can exceed chunk_size by roughly overlap due to tail prepend
        assert len(chunk) <= chunk_size + overlap + 5


def test_overlap_prepends_tail_of_previous_chunk():
    # Use paragraph-separated content so chunks are predictable
    text = "A" * 80 + "\n\n" + "B" * 80
    result = split_text(text, chunk_size=100, overlap=20)
    assert len(result) == 2
    # Second chunk must begin with tail of first
    assert result[1].startswith(result[0][-20:].strip())


def test_chunks_prefer_paragraph_boundary():
    text = "Para one.\n\nPara two.\n\nPara three."
    result = split_text(text, chunk_size=50, overlap=5)
    # Each paragraph is ~10 chars — all fit in one chunk
    assert len(result) == 1


def test_overlap_equal_to_chunk_size_raises():
    with pytest.raises(ValueError, match="overlap"):
        split_text("some text", chunk_size=50, overlap=50)


def test_overlap_greater_than_chunk_size_raises():
    with pytest.raises(ValueError, match="overlap"):
        split_text("some text", chunk_size=50, overlap=60)


def test_result_strips_whitespace():
    text = "  Hello world.  "
    result = split_text(text, chunk_size=100, overlap=10)
    for chunk in result:
        assert chunk == chunk.strip()


def test_no_empty_chunks_in_result():
    text = "\n\n".join(["sentence"] * 50)
    result = split_text(text, chunk_size=80, overlap=10)
    assert all(chunk for chunk in result)
