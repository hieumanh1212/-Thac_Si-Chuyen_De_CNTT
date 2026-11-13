"""FAISS index dùng chunk_id làm ID (IndexIDMap2) => khớp 1-1 với BM25.

IndexFlatIP + vector đã chuẩn hóa L2 => inner product = cosine similarity, tìm kiếm
chính xác (exhaustive). Với vài nghìn chunk thì không cần IVF/HNSW.
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


class FaissStore:
    def __init__(self, index: faiss.Index) -> None:
        self.index = index

    @classmethod
    def build(cls, vectors: np.ndarray, ids: list[int]) -> "FaissStore":
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(vectors.shape[1]))
        index.add_with_ids(vectors, np.asarray(ids, dtype=np.int64))
        return cls(index)

    @property
    def size(self) -> int:
        return int(self.index.ntotal)

    def search(self, query_vec: np.ndarray, top_n: int,
               allowed_ids: set[int] | None = None) -> list[tuple[int, float]]:
        if self.size == 0:
            return []
        # Có bộ lọc tài liệu => quét toàn bộ rồi lọc (corpus nhỏ nên vẫn nhanh)
        k = self.size if allowed_ids is not None else min(top_n, self.size)
        scores, ids = self.index.search(np.ascontiguousarray(query_vec, dtype=np.float32), k)
        out: list[tuple[int, float]] = []
        for cid, score in zip(ids[0], scores[0]):
            if cid == -1 or (allowed_ids is not None and int(cid) not in allowed_ids):
                continue
            out.append((int(cid), float(score)))
            if len(out) >= top_n:
                break
        return out

    # faiss.write_index lỗi với đường dẫn có dấu trên Windows => serialize qua numpy
    def save(self, path: str | Path) -> None:
        np.save(Path(path), faiss.serialize_index(self.index))

    @classmethod
    def load(cls, path: str | Path) -> "FaissStore":
        return cls(faiss.deserialize_index(np.load(Path(path))))
