"""Dữ liệu vào/ra của API (Pydantic) - đây là "hợp đồng" giữa backend và frontend."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Mode = Literal["hybrid", "bm25", "faiss"]


class SearchParams(BaseModel):
    """Tham số tìm kiếm; để trống (null) => dùng giá trị mặc định trong configs/config.yaml."""
    strategy: str | None = None
    mode: Mode | None = None
    top_k: int | None = Field(None, ge=1, le=20)
    bm25_top_n: int | None = Field(None, ge=1, le=100)
    faiss_top_n: int | None = Field(None, ge=1, le=100)
    rrf_k: int | None = Field(None, ge=1, le=1000)
    sources: list[str] | None = None


class AskRequest(SearchParams):
    question: str = Field(..., min_length=1, max_length=2000)
    generate: bool = True          # False = chỉ truy xuất, không gọi LLM


class CompareRequest(SearchParams):
    question: str = Field(..., min_length=1, max_length=2000)
    strategies: list[str] | None = None   # null = mọi chiến lược đã có index


class ChunkOut(BaseModel):
    rank: int
    score: float
    bm25_rank: int | None
    faiss_rank: int | None
    chunk_id: int
    text: str
    source: str
    pages: str
    section: str
    chapter: str | None
    strategy: str
    label: str                      # citation đầy đủ của đoạn này
    cited: bool = False             # LLM có trích dẫn đoạn này không


class SourceOut(BaseModel):
    file: str
    metadata: dict[str, Any]
    locations: list[str]            # ["tr.20-21 (5.3. So sánh Hive và Pig)", ...]


class AskResponse(BaseModel):
    question: str
    strategy: str
    mode: str
    answer: str                     # câu trả lời thô (citation dạng [n])
    answer_rendered: str            # citation đã đổi thành [nguồn, mục, trang]
    llm_error: str | None = None    # lỗi khi gọi LLM (vd. chưa có API key) - vẫn trả kết quả truy xuất
    contexts: list[ChunkOut]
    sources_used: list[SourceOut]
    context_tokens: int
    timings_ms: dict[str, float]


class StrategyResult(BaseModel):
    strategy: str
    total_ms: float
    results: list[ChunkOut]


class CompareResponse(BaseModel):
    question: str
    mode: str
    items: list[StrategyResult]


class SourceInfo(BaseModel):
    file: str
    title: str
    short_ref: str


class InfoResponse(BaseModel):
    strategies: list[str]
    modes: list[str]
    defaults: dict[str, Any]
    llm: dict[str, Any]             # provider, model, ready (đã có API key chưa)
    sources: list[SourceInfo]
    n_chunks: dict[str, int] = {}   # số chunk của từng chiến lược
    index_job: str = "idle"         # idle | running | done | error


class StrategyStats(BaseModel):
    strategy: str
    n_chunks: int | None
    avg_chars: float | None
    median_chars: float | None
    min_chars: int | None
    max_chars: int | None
    multi_page_chunks: int | None
    build_seconds: float | None
    params: dict[str, Any] | None
    length_histogram: dict[str, int]


class StatsResponse(BaseModel):
    embedding_model: str | None
    embedding_dim: int | None
    documents: list[str]
    strategies: list[StrategyStats]
