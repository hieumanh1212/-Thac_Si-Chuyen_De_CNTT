from __future__ import annotations

from typing import Any

from rag.chunking.base import BaseChunker
from rag.chunking.fixed_size import FixedSizeChunker
from rag.chunking.recursive import RecursiveChunker
from rag.chunking.semantic import SemanticChunker

STRATEGIES = ("fixed", "recursive", "semantic")


def get_chunker(name: str, cfg: dict[str, Any], embedder: Any = None) -> BaseChunker:
    """Tạo chunker theo tên + tham số trong config.yaml (mục chunking.<name>)."""
    ccfg = cfg.get("chunking", {})
    params = dict(ccfg.get(name, {}))
    params.setdefault("min_chunk_chars", ccfg.get("min_chunk_chars", 50))
    if name == "fixed":
        return FixedSizeChunker(**params)
    if name == "recursive":
        return RecursiveChunker(**params)
    if name == "semantic":
        if embedder is None:
            raise ValueError("Semantic chunking cần embedder")
        return SemanticChunker(embedder=embedder, **params)
    raise ValueError(f"Chiến lược chunking không hợp lệ: {name}. Chọn một trong {STRATEGIES}")


__all__ = ["BaseChunker", "FixedSizeChunker", "RecursiveChunker", "SemanticChunker", "STRATEGIES", "get_chunker"]
