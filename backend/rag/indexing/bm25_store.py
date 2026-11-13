"""BM25 (keyword search) trên CÙNG tập chunk và CÙNG chunk_id với FAISS."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from rag.indexing.tokenizer import get_tokenizer
from rag.schemas import Chunk


class BM25Store:
    def __init__(self, ids: list[int], tokenized: list[list[str]], tokenizer_name: str = "simple",
                 remove_stopwords: bool = True, k1: float = 1.5, b: float = 0.75) -> None:
        self.ids = list(ids)
        self.tokenized = tokenized
        self.tokenizer_name = tokenizer_name
        self.remove_stopwords = remove_stopwords
        self.k1, self.b = k1, b
        self.tokenize = get_tokenizer(tokenizer_name, remove_stopwords)
        # BM25Okapi lỗi chia 0 nếu có document rỗng => thay bằng token giả
        self.bm25 = BM25Okapi([t or ["<empty>"] for t in tokenized], k1=k1, b=b)

    @classmethod
    def build(cls, chunks: list[Chunk], tokenizer_name: str = "simple", remove_stopwords: bool = True,
              k1: float = 1.5, b: float = 0.75) -> "BM25Store":
        tok = get_tokenizer(tokenizer_name, remove_stopwords)
        return cls([c.chunk_id for c in chunks], [tok(c.text) for c in chunks],
                   tokenizer_name, remove_stopwords, k1, b)

    def search(self, query: str, top_n: int, allowed_ids: set[int] | None = None) -> list[tuple[int, float]]:
        q_tokens = self.tokenize(query)
        if not q_tokens or not self.ids:
            return []
        scores = self.bm25.get_scores(q_tokens)
        out: list[tuple[int, float]] = []
        for idx in np.argsort(-scores):
            if scores[idx] <= 0:
                break
            cid = self.ids[idx]
            if allowed_ids is not None and cid not in allowed_ids:
                continue
            out.append((cid, float(scores[idx])))
            if len(out) >= top_n:
                break
        return out

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"ids": self.ids, "tokenized": self.tokenized, "tokenizer_name": self.tokenizer_name,
                         "remove_stopwords": self.remove_stopwords, "k1": self.k1, "b": self.b}, f)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Store":
        with open(path, "rb") as f:
            return cls(**pickle.load(f))
