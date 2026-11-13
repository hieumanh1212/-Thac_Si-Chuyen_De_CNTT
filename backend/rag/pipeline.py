"""RAGPipeline: điểm vào duy nhất cho app, notebook và script đánh giá.

    question -> HybridRetriever (BM25 + FAISS -> RRF) -> context -> LLM -> answer + citation
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from rag.chunking import STRATEGIES
from rag.generation.context_builder import build_context, cited_numbers, count_tokens, render_citations
from rag.generation.llm_client import BaseLLM, get_llm
from rag.generation.prompts import NO_CONTEXT_ANSWER, SYSTEM_PROMPT, USER_TEMPLATE
from rag.indexing.build_index import CHUNKS_FILE, index_dir
from rag.indexing.embedder import Embedder
from rag.retrieval.hybrid import HybridRetriever, SearchResult
from rag.schemas import RetrievedChunk


@dataclass
class RAGAnswer:
    question: str
    answer: str                         # câu trả lời thô của LLM (citation dạng [n])
    answer_rendered: str                # citation đã đổi thành [nguồn, mục, trang]
    contexts: list[RetrievedChunk]
    labels: list[str]                   # labels[i] = citation của đoạn [i+1]
    cited: list[int]                    # các số [n] mà LLM đã trích dẫn
    context_tokens: int
    strategy: str
    mode: str
    timings_ms: dict[str, float] = field(default_factory=dict)


class RAGPipeline:
    def __init__(self, cfg: dict[str, Any], embedder: Embedder | None = None, llm: BaseLLM | None = None) -> None:
        self.cfg = cfg
        self._embedder = embedder
        self._llm = llm
        self._retrievers: dict[str, HybridRetriever] = {}

    # ---- khởi tạo lười để app mở nhanh và chạy được khi chưa có API key ----
    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder.from_config(self.cfg)
        return self._embedder

    @property
    def llm(self) -> BaseLLM:
        if self._llm is None:
            self._llm = get_llm(self.cfg)
        return self._llm

    def reset(self) -> None:
        """Bỏ các index đang nạp trong RAM (gọi sau khi dựng lại index)."""
        self._retrievers.clear()

    def available_strategies(self) -> list[str]:
        return [s for s in STRATEGIES if (index_dir(self.cfg, s) / CHUNKS_FILE).exists()]

    def retriever(self, strategy: str | None = None) -> HybridRetriever:
        strategy = strategy or self.cfg["retrieval"].get("default_strategy", "recursive")
        if strategy not in self._retrievers:
            self._retrievers[strategy] = HybridRetriever.load(strategy, self.cfg, self.embedder)
        return self._retrievers[strategy]

    # ---- API chính ----
    def retrieve(self, question: str, strategy: str | None = None, **search_kwargs: Any) -> SearchResult:
        return self.retriever(strategy).search(question, **search_kwargs)

    def ask(self, question: str, strategy: str | None = None, generate: bool = True,
            **search_kwargs: Any) -> RAGAnswer:
        sr = self.retrieve(question, strategy, **search_kwargs)
        template = self.cfg.get("citation", {}).get("template", "{short_ref}, {section}, trang {pages}")
        context, labels = build_context(sr.results, template)
        timings = {f"retrieval_{k}": v for k, v in sr.timings_ms.items()}

        if not sr.results:
            answer = NO_CONTEXT_ANSWER
        elif not generate:
            answer = ""
        else:
            t = time.perf_counter()
            answer = self.llm.generate(SYSTEM_PROMPT, USER_TEMPLATE.format(context=context, question=question))
            timings["llm"] = (time.perf_counter() - t) * 1000

        return RAGAnswer(
            question=question,
            answer=answer,
            answer_rendered=render_citations(answer, labels),
            contexts=sr.results,
            labels=labels,
            cited=cited_numbers(answer),
            context_tokens=count_tokens(context) if context else 0,
            strategy=sr.strategy,
            mode=sr.mode,
            timings_ms=timings,
        )
