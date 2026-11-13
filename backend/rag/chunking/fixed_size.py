"""Fixed-size chunking (baseline): cửa sổ trượt cố định số ký tự + overlap.

Chỉ "lùi" điểm cắt về khoảng trắng gần nhất để không cắt đôi một từ; ngoài ra
không quan tâm câu/đoạn/mục => thường cắt ngang câu, dùng làm baseline so sánh.
"""

from __future__ import annotations

from rag.chunking.base import BaseChunker, Span


def _last_space(text: str, lo: int, hi: int) -> int:
    for i in range(hi - 1, lo - 1, -1):
        if text[i].isspace():
            return i
    return -1


def _next_space(text: str, lo: int, hi: int) -> int:
    for i in range(lo, hi):
        if text[i].isspace():
            return i
    return -1


class FixedSizeChunker(BaseChunker):
    name = "fixed"

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100, **kwargs) -> None:
        super().__init__(**kwargs)
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("Cần 0 <= chunk_overlap < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def params(self) -> dict:
        return {**super().params(), "chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap}

    def split_range(self, text: str, start: int, end: int) -> list[Span]:
        spans: list[Span] = []
        pos = start
        while pos < end:
            stop = min(pos + self.chunk_size, end)
            if stop < end:
                ws = _last_space(text, pos + self.chunk_size // 2, stop)
                if ws != -1:
                    stop = ws
            spans.append((pos, stop))
            if stop >= end:
                break
            nxt = stop - self.chunk_overlap
            if self.chunk_overlap:
                ws = _next_space(text, nxt, stop)       # bắt đầu overlap ở đầu một từ
                if ws != -1:
                    nxt = ws + 1
            pos = max(nxt, pos + 1)                     # luôn tiến lên, tránh lặp vô hạn
        return spans
