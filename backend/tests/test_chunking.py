import numpy as np
import pytest

from rag.chunking import FixedSizeChunker, RecursiveChunker, SemanticChunker
from rag.chunking.semantic import sentence_spans
from rag.schemas import Document


def make_doc() -> Document:
    page1 = "1 Introduction\n\n" + " ".join(f"Sentence number {i} talks about cats." for i in range(30))
    page2 = "2 Method\n\n" + " ".join(f"Sentence number {i} explains the dog model." for i in range(30))
    text = page1 + "\n\n" + page2
    p2 = len(page1) + 2
    return Document(
        source="doc.pdf", text=text,
        page_spans=[(0, len(page1), 1), (p2, len(text), 2)],
        section_spans=[(0, "1 Introduction"), (p2, "2 Method")],
        meta={"title": "Doc", "short_ref": "Doc 2024"},
    )


def test_fixed_size_respects_size_and_overlap():
    doc = make_doc()
    chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=40, respect_sections=False, min_chunk_chars=1)
    chunks = chunker.chunk_documents([doc])
    assert all(len(c.text) <= 200 for c in chunks)
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))
    # có overlap: chunk sau bắt đầu trước khi chunk trước kết thúc
    assert any(b.char_start < a.char_end for a, b in zip(chunks, chunks[1:]))
    # không cắt giữa từ
    assert all(doc.text[c.char_start - 1].isspace() for c in chunks if c.char_start > 0)


def test_recursive_respects_sections_and_pages():
    doc = make_doc()
    chunks = RecursiveChunker(chunk_size=300, chunk_overlap=0, min_chunk_chars=1).chunk_documents([doc])
    assert all(len(c.text) <= 300 for c in chunks)
    for c in chunks:
        assert c.section in {"1 Introduction", "2 Method"}
        assert c.page == c.end_page                          # respect_sections => không vắt 2 mục/trang
        expected_page = 1 if c.section == "1 Introduction" else 2
        assert c.page == expected_page
        assert c.metadata["short_ref"] == "Doc 2024"


def test_recursive_merged_chunk_ends_at_sentence():
    doc = make_doc()
    chunks = RecursiveChunker(chunk_size=300, chunk_overlap=0, min_chunk_chars=1).chunk_documents([doc])
    body = [c for c in chunks if len(c.text) > 50]
    assert all(c.text.endswith(".") for c in body)


def test_sentence_spans_skip_abbreviations():
    text = "Results by Smith et al. 2020 are good. Next sentence here."
    spans = sentence_spans(text, 0, len(text))
    assert [text[s:e].strip() for s, e in spans] == ["Results by Smith et al. 2020 are good.", "Next sentence here."]


class FakeEmbedder:
    """Vector theo chủ đề: câu có 'cats' -> [1,0], còn lại -> [0,1]."""

    def encode_passages(self, texts, show_progress=False):
        vecs = np.array([[1.0, 0.0] if "cats" in t and "dog" not in t else [0.0, 1.0] for t in texts], dtype=np.float32)
        return vecs


def test_semantic_breaks_on_topic_change():
    page = " ".join(f"Sentence {i} about cats." for i in range(10)) + " " + \
           " ".join(f"Sentence {i} about the dog." for i in range(10))
    doc = Document(source="d.pdf", text=page, page_spans=[(0, len(page), 1)], section_spans=[])
    chunker = SemanticChunker(FakeEmbedder(), breakpoint_percentile=90, buffer_size=0,
                              min_chars=10, max_chars=5000, respect_sections=False, min_chunk_chars=1)
    chunks = chunker.chunk_documents([doc])
    assert len(chunks) == 2
    assert "dog" not in chunks[0].text and "cats" not in chunks[1].text


def test_invalid_overlap():
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=100, chunk_overlap=100)
