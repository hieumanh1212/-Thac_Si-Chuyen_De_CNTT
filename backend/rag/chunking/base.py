"""Lớp nền cho mọi chiến lược chunking.

Mỗi chiến lược chỉ cần cài đặt `split_range(text, start, end)` trả về các khoảng
ký tự (start, end) TUYỆT ĐỐI trong `text`. Lớp nền lo phần còn lại: chia theo mục
(nếu respect_sections), gán trang/mục/metadata và đánh chunk_id.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag.schemas import Chunk, Document

Span = tuple[int, int]


def strip_span(text: str, start: int, end: int) -> Span:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


class BaseChunker(ABC):
    name: str = "base"

    def __init__(self, respect_sections: bool = True, min_chunk_chars: int = 50) -> None:
        self.respect_sections = respect_sections
        self.min_chunk_chars = min_chunk_chars

    @abstractmethod
    def split_range(self, text: str, start: int, end: int) -> list[Span]:
        """Chia text[start:end] thành các span tuyệt đối."""

    def prepare(self, doc: Document, blocks: list[Span]) -> None:
        """Hook chạy một lần cho mỗi tài liệu trước khi chia (semantic dùng để
        tính embedding câu và ngưỡng trên toàn tài liệu)."""

    def params(self) -> dict:
        return {"respect_sections": self.respect_sections, "min_chunk_chars": self.min_chunk_chars}

    def chunk_document(self, doc: Document, start_id: int = 0) -> list[Chunk]:
        blocks = doc.section_blocks() if self.respect_sections else [(0, len(doc.text))]
        self.prepare(doc, blocks)

        chunks: list[Chunk] = []
        for block_start, block_end in blocks:
            for s, e in self.split_range(doc.text, block_start, block_end):
                s, e = strip_span(doc.text, s, e)
                if e - s < self.min_chunk_chars:
                    continue
                pages = doc.pages_for_span(s, e)
                chunks.append(Chunk(
                    chunk_id=start_id + len(chunks),
                    text=doc.text[s:e],
                    source=doc.source,
                    page=pages[0],
                    end_page=pages[-1],
                    section=doc.section_at(s),
                    strategy=self.name,
                    char_start=s,
                    char_end=e,
                    metadata={**doc.meta, "chapter": doc.chapter_at(s), "sections": doc.sections_in(s, e)},
                ))
        return chunks

    def chunk_documents(self, docs: list[Document]) -> list[Chunk]:
        """chunk_id liên tục 0..N-1 trên toàn corpus => dùng chung cho BM25 & FAISS."""
        all_chunks: list[Chunk] = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc, start_id=len(all_chunks)))
        return all_chunks
