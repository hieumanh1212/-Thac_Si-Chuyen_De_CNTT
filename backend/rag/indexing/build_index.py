"""Dựng index cho từng chiến lược chunking.

Chạy:  python -m rag.indexing.build_index --strategy all
Kết quả: data/indexes/<strategy>/{chunks.jsonl, faiss.npy, bm25.pkl, meta.json}
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from rag.chunking import STRATEGIES, get_chunker
from rag.config import load_config, resolve_path, setup_logging
from rag.indexing.bm25_store import BM25Store
from rag.indexing.embedder import Embedder
from rag.indexing.faiss_store import FaissStore
from rag.ingest import load_corpus
from rag.schemas import Chunk, Document

logger = logging.getLogger(__name__)

CHUNKS_FILE, FAISS_FILE, BM25_FILE, META_FILE = "chunks.jsonl", "faiss.npy", "bm25.pkl", "meta.json"


def index_dir(cfg: dict[str, Any], strategy: str) -> Path:
    return resolve_path(cfg["paths"]["index_dir"]) / strategy


def chunk_stats(chunks: list[Chunk]) -> dict[str, Any]:
    lengths = [len(c.text) for c in chunks] or [0]
    return {
        "n_chunks": len(chunks),
        "avg_chars": round(statistics.mean(lengths), 1),
        "median_chars": statistics.median(lengths),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "multi_page_chunks": sum(c.end_page > c.page for c in chunks),
    }


def build_strategy(strategy: str, docs: list[Document], cfg: dict[str, Any], embedder: Embedder) -> dict[str, Any]:
    t0 = time.perf_counter()
    chunker = get_chunker(strategy, cfg, embedder)
    chunks = chunker.chunk_documents(docs)
    if not chunks:
        raise RuntimeError(f"[{strategy}] Không tạo được chunk nào - kiểm tra lại PDF.")
    t_chunk = time.perf_counter() - t0

    vectors = embedder.encode_passages([c.text for c in chunks], show_progress=True)
    faiss_store = FaissStore.build(vectors, [c.chunk_id for c in chunks])
    bm = cfg.get("bm25", {})
    bm25_store = BM25Store.build(chunks, bm.get("tokenizer", "simple"), bm.get("remove_stopwords", True),
                                 bm.get("k1", 1.5), bm.get("b", 0.75))

    out = index_dir(cfg, strategy)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / CHUNKS_FILE, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
    faiss_store.save(out / FAISS_FILE)
    bm25_store.save(out / BM25_FILE)

    meta = {
        "strategy": strategy,
        "chunker_params": chunker.params(),
        "embedding_model": embedder.model_name,
        "embedding_dim": embedder.dim,
        "bm25": bm,
        "documents": [d.source for d in docs],
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "chunking_seconds": round(t_chunk, 2),
        "total_build_seconds": round(time.perf_counter() - t0, 2),
        **chunk_stats(chunks),
    }
    with open(out / META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def build_indexes(cfg: dict[str, Any], strategies: list[str], embedder: Embedder | None = None) -> list[dict[str, Any]]:
    docs = load_corpus(cfg)
    embedder = embedder or Embedder.from_config(cfg)
    results = []
    for strategy in strategies:
        logger.info("=== Dựng index: %s ===", strategy)
        meta = build_strategy(strategy, docs, cfg, embedder)
        logger.info("[%s] %d chunk, dài TB %.0f ký tự, %.1fs", strategy, meta["n_chunks"],
                    meta["avg_chars"], meta["total_build_seconds"])
        results.append(meta)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Dựng index BM25 + FAISS cho các chiến lược chunking")
    parser.add_argument("--strategy", default="all", help=f"all hoặc một trong {STRATEGIES}")
    parser.add_argument("--config", default=None, help="Đường dẫn config.yaml")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    strategies = cfg["chunking"].get("strategies", list(STRATEGIES)) if args.strategy == "all" else [args.strategy]
    results = build_indexes(cfg, strategies)

    print("\n{:<10} {:>8} {:>10} {:>8} {:>8} {:>8}".format("strategy", "chunks", "avg_chars", "min", "max", "time_s"))
    for m in results:
        print("{:<10} {:>8} {:>10} {:>8} {:>8} {:>8}".format(
            m["strategy"], m["n_chunks"], m["avg_chars"], m["min_chars"], m["max_chars"], m["total_build_seconds"]))


if __name__ == "__main__":
    main()
