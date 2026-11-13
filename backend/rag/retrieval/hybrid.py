"""HybridRetriever: BM25 Top-N + FAISS Top-N -> RRF -> Top-K.

Hỗ trợ 3 chế độ để so sánh trong báo cáo: "bm25", "faiss", "hybrid".
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from rag.indexing.bm25_store import BM25Store
from rag.indexing.build_index import BM25_FILE, CHUNKS_FILE, FAISS_FILE, META_FILE, index_dir
from rag.indexing.embedder import Embedder
from rag.indexing.faiss_store import FaissStore
from rag.retrieval.rrf import reciprocal_rank_fusion
from rag.schemas import Chunk, RetrievedChunk

MODES = ("hybrid", "bm25", "faiss")


@dataclass
class SearchResult:
    query: str
    strategy: str
    mode: str
    results: list[RetrievedChunk]
    timings_ms: dict[str, float] = field(default_factory=dict)
    bm25_candidates: list[tuple[int, float]] = field(default_factory=list)
    faiss_candidates: list[tuple[int, float]] = field(default_factory=list)


class HybridRetriever:
    def __init__(self, strategy: str, chunks: list[Chunk], faiss_store: FaissStore, bm25_store: BM25Store,
                 embedder: Embedder, retrieval_cfg: dict[str, Any], meta: dict[str, Any] | None = None) -> None:
        self.strategy = strategy
        self.chunks: dict[int, Chunk] = {c.chunk_id: c for c in chunks}
        self.faiss = faiss_store
        self.bm25 = bm25_store
        self.embedder = embedder
        self.cfg = retrieval_cfg
        self.meta = meta or {}
        # Kiểm tra ràng buộc của đề: BM25 & FAISS phải cùng tập chunk_id
        if set(self.bm25.ids) != set(self.chunks) or self.faiss.size != len(self.chunks):
            raise ValueError(f"[{strategy}] BM25/FAISS/chunks không khớp chunk_id - hãy build lại index.")

    @classmethod
    def load(cls, strategy: str, cfg: dict[str, Any], embedder: Embedder) -> "HybridRetriever":
        d = index_dir(cfg, strategy)
        if not (d / CHUNKS_FILE).exists():
            raise FileNotFoundError(f"Chưa có index cho '{strategy}'. Chạy: python -m rag.indexing.build_index")
        with open(d / META_FILE, encoding="utf-8") as f:
            meta = json.load(f)
        if meta["embedding_model"] != embedder.model_name:
            raise ValueError(f"Index '{strategy}' dựng bằng {meta['embedding_model']} nhưng đang dùng "
                             f"{embedder.model_name}. Hãy build lại index.")
        with open(d / CHUNKS_FILE, encoding="utf-8") as f:
            chunks = [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]
        return cls(strategy, chunks, FaissStore.load(d / FAISS_FILE), BM25Store.load(d / BM25_FILE),
                   embedder, cfg.get("retrieval", {}), meta)

    # ---- tiện ích ----
    def sources(self) -> list[str]:
        return sorted({c.source for c in self.chunks.values()})

    def _allowed_ids(self, sources: list[str] | None) -> set[int] | None:
        if not sources:
            return None
        wanted = set(sources)
        return {cid for cid, c in self.chunks.items() if c.source in wanted}

    # ---- tìm kiếm ----
    def search(self, query: str, mode: str | None = None, top_k: int | None = None,
               bm25_top_n: int | None = None, faiss_top_n: int | None = None, rrf_k: int | None = None,
               sources: list[str] | None = None) -> SearchResult:
        mode = mode or self.cfg.get("default_mode", "hybrid")
        if mode not in MODES:
            raise ValueError(f"mode phải là một trong {MODES}")
        top_k = top_k or self.cfg.get("top_k", 5)
        bm25_n = max(bm25_top_n or self.cfg.get("bm25_top_n", 8), top_k if mode == "bm25" else 0)
        faiss_n = max(faiss_top_n or self.cfg.get("faiss_top_n", 8), top_k if mode == "faiss" else 0)
        rrf_k = rrf_k or self.cfg.get("rrf_k", 60)
        allowed = self._allowed_ids(sources)
        timings: dict[str, float] = {}
        t_start = time.perf_counter()

        bm25_hits: list[tuple[int, float]] = []
        faiss_hits: list[tuple[int, float]] = []
        if mode in ("bm25", "hybrid"):
            t = time.perf_counter()
            bm25_hits = self.bm25.search(query, bm25_n, allowed)
            timings["bm25"] = (time.perf_counter() - t) * 1000
        if mode in ("faiss", "hybrid"):
            t = time.perf_counter()
            qvec = self.embedder.encode_query(query)
            timings["embed_query"] = (time.perf_counter() - t) * 1000
            t = time.perf_counter()
            faiss_hits = self.faiss.search(qvec, faiss_n, allowed)
            timings["faiss"] = (time.perf_counter() - t) * 1000

        bm25_rank = {cid: r for r, (cid, _) in enumerate(bm25_hits, start=1)}
        faiss_rank = {cid: r for r, (cid, _) in enumerate(faiss_hits, start=1)}
        bm25_score, faiss_score = dict(bm25_hits), dict(faiss_hits)

        if mode == "hybrid":
            t = time.perf_counter()
            fused = reciprocal_rank_fusion({"bm25": [c for c, _ in bm25_hits],
                                            "faiss": [c for c, _ in faiss_hits]}, k=rrf_k)
            timings["rrf"] = (time.perf_counter() - t) * 1000
            ranked = [(cid, score) for cid, score, _ in fused[:top_k]]
        else:
            ranked = (bm25_hits if mode == "bm25" else faiss_hits)[:top_k]

        results = [
            RetrievedChunk(chunk=self.chunks[cid], score=score, rank=i,
                           bm25_rank=bm25_rank.get(cid), faiss_rank=faiss_rank.get(cid),
                           bm25_score=bm25_score.get(cid), faiss_score=faiss_score.get(cid))
            for i, (cid, score) in enumerate(ranked, start=1)
        ]
        timings["total"] = (time.perf_counter() - t_start) * 1000
        return SearchResult(query, self.strategy, mode, results, timings, bm25_hits, faiss_hits)
