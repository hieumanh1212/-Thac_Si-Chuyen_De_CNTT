"""Bọc SentenceTransformer: dùng CÙNG một mô hình cho chunk và câu hỏi."""

from __future__ import annotations

from typing import Any

import numpy as np


class Embedder:
    def __init__(self, model_name: str, query_prefix: str = "", passage_prefix: str = "",
                 batch_size: int = 32, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # import chậm => để trong hàm

        self.model_name = model_name
        self.query_prefix = query_prefix or ""
        self.passage_prefix = passage_prefix or ""
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Embedder":
        e = cfg["embedding"]
        return cls(e["model_name"], e.get("query_prefix", ""), e.get("passage_prefix", ""),
                   e.get("batch_size", 32), e.get("device"))

    @property
    def dim(self) -> int:
        # sentence-transformers >= 6 đổi tên hàm; giữ tương thích bản cũ
        getter = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        return int(getter())

    def _encode(self, texts: list[str], show_progress: bool) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=show_progress)
        return np.asarray(vecs, dtype=np.float32)

    def encode_passages(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        return self._encode([self.passage_prefix + t for t in texts], show_progress)

    def encode_query(self, query: str) -> np.ndarray:
        return self._encode([self.query_prefix + query], False)  # shape (1, dim)
