"""Backend REST API cho Hybrid RAG.

Chạy (từ thư mục backend):
    uvicorn api.main:app --port 8000 --reload
Tài liệu API tương tác: http://localhost:8000/docs

Các endpoint:
    GET  /api/health    kiểm tra server sống
    GET  /api/info      danh sách chiến lược, chế độ, tham số mặc định, tài liệu
    POST /api/ask       hỏi đáp (truy xuất + LLM)
    POST /api/compare   so sánh kết quả truy xuất giữa các chiến lược chunking
    GET  /api/stats     thống kê index của từng chiến lược
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.documents import IndexJob, create_router
from api.schemas import (AskRequest, AskResponse, ChunkOut, CompareRequest, CompareResponse, InfoResponse,
                         SearchParams, SourceInfo, SourceOut, StatsResponse, StrategyResult, StrategyStats)
from rag.config import load_config, setup_logging
from rag.generation.context_builder import citation_label, format_pages
from rag.pipeline import RAGAnswer, RAGPipeline
from rag.retrieval.hybrid import MODES
from rag.schemas import RetrievedChunk

logger = logging.getLogger("api")

META_FIELDS = ["title", "authors", "year", "venue", "document_type", "doi_url", "source_url", "course",
               "short_ref", "number_symbol", "issue_date", "issuing_authority"]
HIST_BINS = list(range(0, 2001, 200))     # 0-200, 200-400, ..., 1800-2000, >2000


# ------------------------------ chuyển đổi dữ liệu ------------------------------
def chunk_out(r: RetrievedChunk, label: str, cited: bool = False) -> ChunkOut:
    c = r.chunk
    chapter = c.metadata.get("chapter")
    return ChunkOut(rank=r.rank, score=r.score, bm25_rank=r.bm25_rank, faiss_rank=r.faiss_rank,
                    chunk_id=c.chunk_id, text=c.text, source=c.source, pages=format_pages(c),
                    section=c.section or "", chapter=chapter if chapter and chapter != c.section else None,
                    strategy=c.strategy, label=label, cited=cited)


def sources_used(ans: RAGAnswer) -> list[SourceOut]:
    """Tài liệu đã dùng: ưu tiên các đoạn được LLM trích dẫn, không có thì lấy toàn bộ context."""
    used = [ans.contexts[i - 1] for i in ans.cited if 1 <= i <= len(ans.contexts)] or ans.contexts
    rows: dict[str, SourceOut] = {}
    for r in used:
        c = r.chunk
        row = rows.setdefault(c.source, SourceOut(
            file=c.source, locations=[],
            metadata={f: c.metadata[f] for f in META_FIELDS if c.metadata.get(f) not in (None, "")}))
        loc = f"tr.{format_pages(c)} ({c.section or '-'})"
        if loc not in row.locations:
            row.locations.append(loc)
    return list(rows.values())


def ask_response(ans: RAGAnswer, llm_error: str | None = None) -> AskResponse:
    cited = set(ans.cited)
    return AskResponse(
        question=ans.question, strategy=ans.strategy, mode=ans.mode,
        answer=ans.answer, answer_rendered=ans.answer_rendered, llm_error=llm_error,
        contexts=[chunk_out(r, label, i in cited) for i, (r, label) in enumerate(zip(ans.contexts, ans.labels), 1)],
        sources_used=sources_used(ans) if ans.contexts else [],
        context_tokens=ans.context_tokens, timings_ms={k: round(v, 2) for k, v in ans.timings_ms.items()},
    )


def length_histogram(lengths: list[int]) -> dict[str, int]:
    hist = {f"{a}-{b}": 0 for a, b in zip(HIST_BINS, HIST_BINS[1:])}
    hist[f">{HIST_BINS[-1]}"] = 0
    keys = list(hist)
    for n in lengths:
        idx = min(n // (HIST_BINS[1] - HIST_BINS[0]), len(keys) - 1)
        hist[keys[idx]] += 1
    return hist


def llm_ready(llm_cfg: dict[str, Any]) -> bool:
    """Đã cấu hình API key cho LLM chưa (chỉ kiểm tra có key, không gọi thử)."""
    provider = llm_cfg.get("provider", "anthropic")
    if provider == "anthropic":
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    base_url = str(llm_cfg.get("base_url") or os.getenv("LLM_BASE_URL") or "")
    is_local = any(h in base_url for h in ("localhost", "127.0.0.1"))    # Ollama không cần key
    return is_local or bool(os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"))


# ------------------------------ ứng dụng ------------------------------
def create_app(pipeline: RAGPipeline | None = None) -> FastAPI:
    """Tạo app. Truyền `pipeline` giả vào khi viết test để không phải nạp mô hình thật."""
    state: dict[str, Any] = {"pipe": pipeline}
    lock = threading.Lock()   # embedder/LLM client không đảm bảo an toàn đa luồng

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if state["pipe"] is None:
            setup_logging()
            pipe = RAGPipeline(load_config())
            logger.info("Đang nạp mô hình embedding...")
            _ = pipe.embedder                       # nạp sẵn để request đầu tiên không bị chậm
            state["pipe"] = pipe
            logger.info("Sẵn sàng. Chiến lược có index: %s", pipe.available_strategies() or "chưa có")
        yield

    app = FastAPI(title="Hybrid RAG API", version="1.0.0", lifespan=lifespan,
                  description="BM25 + FAISS + RRF với nhiều chiến lược chunking.")

    origins = os.getenv("FRONTEND_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in origins if o.strip()],
                       allow_methods=["GET", "POST", "PUT", "DELETE"], allow_headers=["*"])

    def pipe() -> RAGPipeline:
        return state["pipe"]

    job = IndexJob()
    app.include_router(create_router(pipe, lock, job))

    def check_strategy(strategy: str | None) -> str:
        available = pipe().available_strategies()
        if not available:
            raise HTTPException(409, "Chưa có index. Chạy: python -m rag.indexing.build_index --strategy all")
        strategy = strategy or pipe().cfg["retrieval"].get("default_strategy", available[0])
        if strategy not in available:
            raise HTTPException(400, f"Chiến lược '{strategy}' chưa có index. Có sẵn: {available}")
        return strategy

    def search_kwargs(p: SearchParams) -> dict[str, Any]:
        return dict(mode=p.mode, top_k=p.top_k, bm25_top_n=p.bm25_top_n, faiss_top_n=p.faiss_top_n,
                    rrf_k=p.rrf_k, sources=p.sources or None)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "ready": pipe() is not None}

    @app.get("/api/info", response_model=InfoResponse)
    def info() -> InfoResponse:
        p = pipe()
        strategies = p.available_strategies()
        r = p.cfg.get("retrieval", {})
        sources: list[SourceInfo] = []
        n_chunks: dict[str, int] = {}
        if strategies:
            with lock:
                retrievers = {s: p.retriever(s) for s in strategies}
            n_chunks = {s: len(rv.chunks) for s, rv in retrievers.items()}
            seen: dict[str, SourceInfo] = {}
            for c in retrievers[strategies[0]].chunks.values():
                if c.source not in seen:
                    title = c.metadata.get("title") or c.source
                    seen[c.source] = SourceInfo(file=c.source, title=title,
                                                short_ref=c.metadata.get("short_ref") or title)
            sources = sorted(seen.values(), key=lambda s: s.file)
        llm_cfg = p.cfg.get("llm", {})
        return InfoResponse(
            strategies=strategies, modes=list(MODES), sources=sources, n_chunks=n_chunks,
            index_job=job.snapshot()["status"],
            defaults={"strategy": r.get("default_strategy", strategies[0] if strategies else None),
                      "mode": r.get("default_mode", "hybrid"), "top_k": r.get("top_k", 5),
                      "bm25_top_n": r.get("bm25_top_n", 8), "faiss_top_n": r.get("faiss_top_n", 8),
                      "rrf_k": r.get("rrf_k", 60)},
            llm={"provider": llm_cfg.get("provider"), "model": llm_cfg.get("model"), "ready": llm_ready(llm_cfg)},
        )

    @app.post("/api/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> AskResponse:
        strategy = check_strategy(req.strategy)
        question = req.question.strip()
        if not question:
            raise HTTPException(422, "Câu hỏi trống.")
        with lock:
            try:
                return ask_response(pipe().ask(question, strategy=strategy, generate=req.generate,
                                               **search_kwargs(req)))
            except HTTPException:
                raise
            except Exception as exc:
                if not req.generate:
                    logger.exception("Lỗi truy xuất")
                    raise HTTPException(500, f"Lỗi truy xuất: {exc}") from exc
                # LLM lỗi (thiếu API key, hết quota, mạng...) => vẫn trả kết quả truy xuất kèm thông báo
                logger.warning("Lỗi LLM: %s", exc)
                ans = pipe().ask(question, strategy=strategy, generate=False, **search_kwargs(req))
                return ask_response(ans, llm_error=f"{type(exc).__name__}: {exc}")

    @app.post("/api/compare", response_model=CompareResponse)
    def compare(req: CompareRequest) -> CompareResponse:
        check_strategy(None)
        available = pipe().available_strategies()
        wanted = req.strategies or available
        bad = [s for s in wanted if s not in available]
        if bad:
            raise HTTPException(400, f"Chiến lược chưa có index: {bad}")
        template = pipe().cfg.get("citation", {}).get("template", "{short_ref}, {section}, trang {pages}")
        items: list[StrategyResult] = []
        mode = req.mode
        with lock:
            for strat in wanted:
                sr = pipe().retrieve(req.question.strip(), strategy=strat, **search_kwargs(req))
                mode = sr.mode
                items.append(StrategyResult(
                    strategy=strat, total_ms=round(sr.timings_ms.get("total", 0.0), 2),
                    results=[chunk_out(r, citation_label(r.chunk, template)) for r in sr.results]))
        return CompareResponse(question=req.question, mode=mode or "hybrid", items=items)

    @app.get("/api/stats", response_model=StatsResponse)
    def stats() -> StatsResponse:
        check_strategy(None)
        rows: list[StrategyStats] = []
        meta0: dict[str, Any] = {}
        with lock:
            for strat in pipe().available_strategies():
                rv = pipe().retriever(strat)
                m = rv.meta
                meta0 = meta0 or m
                rows.append(StrategyStats(
                    strategy=strat, n_chunks=m.get("n_chunks"), avg_chars=m.get("avg_chars"),
                    median_chars=m.get("median_chars"), min_chars=m.get("min_chars"),
                    max_chars=m.get("max_chars"), multi_page_chunks=m.get("multi_page_chunks"),
                    build_seconds=m.get("total_build_seconds"), params=m.get("chunker_params"),
                    length_histogram=length_histogram([len(c.text) for c in rv.chunks.values()])))
        return StatsResponse(embedding_model=meta0.get("embedding_model"), embedding_dim=meta0.get("embedding_dim"),
                             documents=list(meta0.get("documents", [])), strategies=rows)

    return app


app = create_app()
