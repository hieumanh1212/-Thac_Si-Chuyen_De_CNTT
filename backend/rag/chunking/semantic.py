"""Semantic chunking: cắt tại các điểm "đổi chủ đề" giữa các câu liên tiếp.

Thuật toán:
1. Tách câu (giữ offset).
2. Embedding mỗi câu kèm `buffer_size` câu trước/sau (giảm nhiễu của câu ngắn).
3. Khoảng cách cosine giữa hai câu liên tiếp: d_i = 1 - cos(e_i, e_{i+1}).
4. Ngưỡng = phân vị `breakpoint_percentile` của mọi d_i TRONG CẢ TÀI LIỆU
   (vd 90 => chỉ 10% vị trí "đổi chủ đề mạnh nhất" thành điểm cắt).
5. Ràng buộc độ dài: không vượt max_chars (cắt sớm), nhóm < min_chars thì gộp
   vào nhóm kế tiếp, câu đơn lẻ dài hơn max_chars thì cắt cứng theo khoảng trắng.
"""

from __future__ import annotations

import re
from typing import Protocol

import numpy as np

from rag.chunking.base import BaseChunker, Span
from rag.chunking.fixed_size import FixedSizeChunker
from rag.schemas import Document

_SENT_END = re.compile(r"[.!?…]+[\"')\]]*\s+")
_ABBREVIATIONS = {"al", "e.g", "i.e", "fig", "eq", "vs", "etc", "no", "tab", "sec", "ref", "cf", "dr", "mr", "ms", "pp", "vol"}


class _Encoder(Protocol):
    def encode_passages(self, texts: list[str], show_progress: bool = False) -> np.ndarray: ...


def sentence_spans(text: str, start: int, end: int) -> list[Span]:
    """Tách câu trong text[start:end]; luôn tách ở ranh giới đoạn "\\n\\n"."""
    spans: list[Span] = []
    for para_start, para_end in _paragraphs(text, start, end):
        pos = para_start
        for m in _SENT_END.finditer(text, para_start, para_end):
            nxt = m.end()
            if nxt >= para_end:
                break
            following = text[nxt]
            word_before = text[pos:m.start()].split()[-1:] or [""]
            if word_before[0].lower().rstrip(".") in _ABBREVIATIONS:
                continue
            if following.isupper() or following.isdigit() or following in "\"'([":
                spans.append((pos, m.end()))
                pos = nxt
        if pos < para_end:
            spans.append((pos, para_end))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def _paragraphs(text: str, start: int, end: int) -> list[Span]:
    out: list[Span] = []
    pos = start
    while pos < end:
        idx = text.find("\n\n", pos, end)
        stop = end if idx == -1 else idx
        if stop > pos:
            out.append((pos, stop))
        pos = stop + 2 if idx != -1 else end
    return out


class SemanticChunker(BaseChunker):
    name = "semantic"

    def __init__(self, embedder: _Encoder, breakpoint_percentile: float = 90, buffer_size: int = 1,
                 min_chars: int = 200, max_chars: int = 1200, **kwargs) -> None:
        super().__init__(**kwargs)
        self.embedder = embedder
        self.breakpoint_percentile = breakpoint_percentile
        self.buffer_size = buffer_size
        self.min_chars = min_chars
        self.max_chars = max_chars
        self._fallback = FixedSizeChunker(chunk_size=max_chars, chunk_overlap=0)
        # trạng thái tính trong prepare() cho tài liệu hiện tại
        self._sents: list[Span] = []
        self._dist: np.ndarray = np.zeros(0)
        self._threshold: float = 1.0

    def params(self) -> dict:
        return {**super().params(), "breakpoint_percentile": self.breakpoint_percentile,
                "buffer_size": self.buffer_size, "min_chars": self.min_chars, "max_chars": self.max_chars}

    def prepare(self, doc: Document, blocks: list[Span]) -> None:
        self._sents = [s for b in blocks for s in sentence_spans(doc.text, *b)]
        n = len(self._sents)
        if n < 2:
            self._dist, self._threshold = np.zeros(0), 1.0
            return
        b = self.buffer_size
        windows = [doc.text[self._sents[max(0, i - b)][0]:self._sents[min(n - 1, i + b)][1]] for i in range(n)]
        emb = self.embedder.encode_passages(windows)          # đã chuẩn hóa L2
        self._dist = 1.0 - np.sum(emb[:-1] * emb[1:], axis=1)
        self._threshold = float(np.percentile(self._dist, self.breakpoint_percentile))

    def split_range(self, text: str, start: int, end: int) -> list[Span]:
        idx = [i for i, (s, e) in enumerate(self._sents) if s >= start and e <= end]
        if not idx:
            return [(start, end)] if end > start else []

        # (4) cắt theo ngưỡng + (5a) không vượt max_chars
        groups: list[Span] = []
        g_start, g_end = self._sents[idx[0]]
        for prev, i in zip(idx, idx[1:]):
            s, e = self._sents[i]
            consecutive = i == prev + 1
            is_break = consecutive and prev < len(self._dist) and self._dist[prev] > self._threshold
            if is_break or e - g_start > self.max_chars:
                groups.append((g_start, g_end))
                g_start = s
            g_end = e
        groups.append((g_start, g_end))

        # (5b) gộp nhóm quá ngắn vào nhóm kế tiếp nếu không vượt max_chars
        merged: list[Span] = []
        for g in groups:
            if merged and merged[-1][1] - merged[-1][0] < self.min_chars and g[1] - merged[-1][0] <= self.max_chars:
                merged[-1] = (merged[-1][0], g[1])
            else:
                merged.append(g)
        if len(merged) > 1 and merged[-1][1] - merged[-1][0] < self.min_chars \
                and merged[-1][1] - merged[-2][0] <= self.max_chars:
            last = merged.pop()
            merged[-1] = (merged[-1][0], last[1])

        # (5c) câu đơn dài hơn max_chars => cắt cứng
        out: list[Span] = []
        for s, e in merged:
            out.extend(self._fallback.split_range(text, s, e) if e - s > self.max_chars else [(s, e)])
        return out
