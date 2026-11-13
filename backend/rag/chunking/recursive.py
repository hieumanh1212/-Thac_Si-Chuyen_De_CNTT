"""Recursive chunking: chia theo thứ tự ưu tiên đoạn -> dòng -> câu -> mệnh đề -> từ.

Thuật toán (giống RecursiveCharacterTextSplitter, tự cài đặt để giữ offset ký tự):
1. Thử tách bằng separator lớn nhất ("\\n\\n"). Mảnh nào vẫn > chunk_size thì
   đệ quy tách tiếp bằng separator nhỏ hơn.
2. Gộp các mảnh liền kề lại cho tới khi gần chạm chunk_size.
3. Overlap: chunk mới bắt đầu bằng vài mảnh cuối của chunk trước (tổng <= overlap).
"""

from __future__ import annotations

from rag.chunking.base import BaseChunker, Span

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]


def split_by_separator(text: str, start: int, end: int, sep: str) -> list[Span]:
    """Tách text[start:end] theo sep, giữ sep ở cuối mảnh trước."""
    pieces: list[Span] = []
    pos = start
    while True:
        idx = text.find(sep, pos, end)
        if idx == -1:
            break
        cut = idx + len(sep)
        if cut > pos:
            pieces.append((pos, cut))
        pos = cut
    if pos < end:
        pieces.append((pos, end))
    return pieces


class RecursiveChunker(BaseChunker):
    name = "recursive"

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100,
                 separators: list[str] | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        if not 0 <= chunk_overlap < chunk_size:
            raise ValueError("Cần 0 <= chunk_overlap < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or DEFAULT_SEPARATORS

    def params(self) -> dict:
        return {**super().params(), "chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap}

    # ---- bước 1: tách đệ quy thành các mảnh <= chunk_size ----
    def _split(self, text: str, start: int, end: int, seps: list[str]) -> list[Span]:
        if end - start <= self.chunk_size:
            return [(start, end)]
        if not seps:  # hết separator: cắt cứng
            return [(i, min(i + self.chunk_size, end)) for i in range(start, end, self.chunk_size)]
        pieces = split_by_separator(text, start, end, seps[0])
        if len(pieces) <= 1:
            return self._split(text, start, end, seps[1:])
        out: list[Span] = []
        for s, e in pieces:
            out.extend(self._split(text, s, e, seps[1:]) if e - s > self.chunk_size else [(s, e)])
        return out

    # ---- bước 2 + 3: gộp mảnh liền kề, thêm overlap ----
    def _merge(self, pieces: list[Span]) -> list[Span]:
        chunks: list[Span] = []
        cur: list[Span] = []
        for piece in pieces:
            if cur and piece[1] - cur[0][0] > self.chunk_size:
                chunks.append((cur[0][0], cur[-1][1]))
                keep: list[Span] = []
                kept_len = 0
                while cur and kept_len + (cur[-1][1] - cur[-1][0]) <= self.chunk_overlap:
                    kept_len += cur[-1][1] - cur[-1][0]
                    keep.insert(0, cur.pop())
                cur = keep
                while cur and piece[1] - cur[0][0] > self.chunk_size:
                    cur.pop(0)
            cur.append(piece)
        if cur:
            chunks.append((cur[0][0], cur[-1][1]))
        return chunks

    def split_range(self, text: str, start: int, end: int) -> list[Span]:
        return self._merge(self._split(text, start, end, self.separators))
