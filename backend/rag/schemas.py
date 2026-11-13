"""Các cấu trúc dữ liệu dùng chung: Document, Chunk, RetrievedChunk."""

from __future__ import annotations

import bisect
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Document:
    """Một file PDF sau khi đọc từng trang và làm sạch.

    Các trang được nối bằng "\\n\\n" thành `text` để chunk có thể vắt qua ranh giới
    trang, nhưng `page_spans` giữ lại vị trí ký tự của từng trang => mọi chunk vẫn
    truy ngược được về số trang gốc (yêu cầu 3.1 của đề).
    """

    source: str                                  # tên file, vd "paper_a.pdf"
    text: str                                    # nội dung đã làm sạch của cả tài liệu
    page_spans: list[tuple[int, int, int]]       # (char_start, char_end, page_number 1-based)
    section_spans: list[tuple[int, str]]         # (char_offset, tiêu đề mục), tăng dần
    meta: dict[str, Any] = field(default_factory=dict)

    # ---- tra cứu trang ----
    def page_at(self, offset: int) -> int:
        if not self.page_spans:
            return 1
        starts = [s for s, _, _ in self.page_spans]
        i = bisect.bisect_right(starts, offset) - 1
        return self.page_spans[max(i, 0)][2]

    def pages_for_span(self, start: int, end: int) -> list[int]:
        pages = [p for s, e, p in self.page_spans if s < end and e > start]
        return pages or [self.page_at(start)]

    # ---- tra cứu mục ----
    def section_at(self, offset: int) -> str:
        offsets = [o for o, _ in self.section_spans]
        i = bisect.bisect_right(offsets, offset) - 1
        return self.section_spans[i][1] if i >= 0 else ""

    def chapter_at(self, offset: int) -> str:
        """Tiêu đề cấp 1 (chương) bao trùm vị trí offset."""
        from rag.ingest.section_detector import heading_level

        chapter = ""
        for off, title in self.section_spans:
            if off > offset:
                break
            if heading_level(title) == 1:
                chapter = title
        return chapter

    def sections_in(self, start: int, end: int) -> list[str]:
        titles = [self.section_at(start)] if self.section_at(start) else []
        for off, title in self.section_spans:
            if start < off < end and title not in titles:
                titles.append(title)
        return titles

    def section_blocks(self) -> list[tuple[int, int]]:
        """Chia text thành các khối theo ranh giới mục (dùng khi respect_sections=True)."""
        cuts = sorted({0, len(self.text), *(o for o, _ in self.section_spans if 0 < o < len(self.text))})
        return [(a, b) for a, b in zip(cuts, cuts[1:]) if b > a]

    def remove_range(self, start: int, end: int) -> None:
        """Xóa text[start:end] (vd: phần References) và dịch lại vị trí trang/mục phía sau."""
        start, end = max(0, start), min(len(self.text), end)
        if end <= start:
            return
        if end >= len(self.text):            # xóa đến hết => bỏ luôn "\n\n" thừa ở cuối
            start = len(self.text[:start].rstrip())
        shift = end - start

        def move(pos: int) -> int:
            return pos if pos <= start else (start if pos < end else pos - shift)

        self.text = self.text[:start] + self.text[end:]
        spans = [(move(s), move(e), p) for s, e, p in self.page_spans]
        self.page_spans = [(s, e, p) for s, e, p in spans if e > s]
        self.section_spans = [(o - (shift if o >= end else 0), t) for o, t in self.section_spans
                              if not start <= o < end]

    # ---- serialize ----
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Document":
        return cls(
            source=d["source"],
            text=d["text"],
            page_spans=[tuple(x) for x in d["page_spans"]],
            section_spans=[tuple(x) for x in d["section_spans"]],
            meta=d.get("meta", {}),
        )


@dataclass
class Chunk:
    """Đơn vị được index. chunk_id dùng chung cho cả BM25 và FAISS."""

    chunk_id: int
    text: str
    source: str
    page: int                 # trang bắt đầu
    end_page: int             # trang kết thúc (chunk có thể vắt qua 2 trang)
    section: str
    strategy: str
    char_start: int = 0
    char_end: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def pages(self) -> range:
        return range(self.page, self.end_page + 1)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Chunk":
        return cls(**d)


@dataclass
class RetrievedChunk:
    """Một kết quả tìm kiếm kèm điểm và thứ hạng ở từng retriever."""

    chunk: Chunk
    score: float
    rank: int
    bm25_rank: int | None = None
    faiss_rank: int | None = None
    bm25_score: float | None = None
    faiss_score: float | None = None
