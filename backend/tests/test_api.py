"""Test REST API bằng pipeline giả (không nạp mô hình embedding, không gọi LLM thật)."""

from types import SimpleNamespace

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from api.main import create_app, length_histogram  # noqa: E402
from rag.generation.context_builder import cited_numbers, render_citations  # noqa: E402
from rag.pipeline import RAGAnswer  # noqa: E402
from rag.retrieval.hybrid import SearchResult  # noqa: E402
from rag.schemas import Chunk, RetrievedChunk  # noqa: E402

CHUNKS = [
    Chunk(chunk_id=0, text="Hive là kho dữ liệu trên Hadoop.", source="a.pdf", page=20, end_page=21,
          section="5.3 Hive", strategy="recursive",
          metadata={"title": "Báo cáo A", "short_ref": "Nhóm 1, 2026", "chapter": "Chương 5"}),
    Chunk(chunk_id=1, text="Pig dùng ngôn ngữ Pig Latin.", source="a.pdf", page=22, end_page=22,
          section="5.4 Pig", strategy="recursive", metadata={"title": "Báo cáo A", "short_ref": "Nhóm 1, 2026"}),
]


class FakePipe:
    def __init__(self, llm_fails: bool = False):
        self.llm_fails = llm_fails
        self.cfg = {"retrieval": {"default_strategy": "recursive", "top_k": 5},
                    "llm": {"provider": "fake", "model": "fake-1"},
                    "citation": {"template": "{short_ref}, {section}, trang {pages}"}}

    def available_strategies(self):
        return ["recursive", "semantic"]

    def retriever(self, strategy=None):
        return SimpleNamespace(chunks={c.chunk_id: c for c in CHUNKS},
                               meta={"n_chunks": 2, "avg_chars": 30.0, "embedding_model": "e5",
                                     "embedding_dim": 768, "documents": ["a.pdf"]})

    def retrieve(self, question, strategy=None, **kw):
        res = [RetrievedChunk(chunk=c, score=1 / (i + 1), rank=i + 1, bm25_rank=i + 1, faiss_rank=None)
               for i, c in enumerate(CHUNKS)]
        return SearchResult(question, strategy or "recursive", kw.get("mode") or "hybrid", res, {"total": 1.5})

    def ask(self, question, strategy=None, generate=True, **kw):
        if generate and self.llm_fails:
            raise RuntimeError("thiếu API key")
        sr = self.retrieve(question, strategy, **kw)
        labels = ["Nhóm 1, 2026, 5.3 Hive, trang 20-21", "Nhóm 1, 2026, 5.4 Pig, trang 22"]
        answer = "Hive là kho dữ liệu [1]." if generate else ""
        return RAGAnswer(question, answer, render_citations(answer, labels), sr.results, labels,
                         cited_numbers(answer), 42, sr.strategy, sr.mode, {"retrieval_total": 1.5, "llm": 900.0})


@pytest.fixture
def client():
    with TestClient(create_app(FakePipe())) as c:
        yield c


def test_info(client):
    data = client.get("/api/info").json()
    assert data["strategies"] == ["recursive", "semantic"]
    assert data["defaults"]["strategy"] == "recursive"
    assert data["sources"] == [{"file": "a.pdf", "title": "Báo cáo A", "short_ref": "Nhóm 1, 2026"}]


def test_ask_with_citation(client):
    data = client.post("/api/ask", json={"question": "Hive là gì?"}).json()
    assert data["answer_rendered"] == "Hive là kho dữ liệu [Nhóm 1, 2026, 5.3 Hive, trang 20-21]."
    assert [c["cited"] for c in data["contexts"]] == [True, False]
    assert data["contexts"][0]["chapter"] == "Chương 5" and data["contexts"][0]["pages"] == "20-21"
    assert data["sources_used"][0]["locations"] == ["tr.20-21 (5.3 Hive)"]
    assert data["llm_error"] is None


def test_ask_llm_error_still_returns_contexts():
    with TestClient(create_app(FakePipe(llm_fails=True))) as c:
        data = c.post("/api/ask", json={"question": "Hive?"}).json()
    assert "thiếu API key" in data["llm_error"]
    assert data["answer"] == "" and len(data["contexts"]) == 2


def test_validation(client):
    assert client.post("/api/ask", json={"question": ""}).status_code == 422
    assert client.post("/api/ask", json={"question": "x", "mode": "abc"}).status_code == 422
    assert client.post("/api/ask", json={"question": "x", "strategy": "fixed"}).status_code == 400


def test_compare_and_stats(client):
    cmp = client.post("/api/compare", json={"question": "Hive"}).json()
    assert [i["strategy"] for i in cmp["items"]] == ["recursive", "semantic"]
    stats = client.get("/api/stats").json()
    assert stats["embedding_dim"] == 768 and len(stats["strategies"]) == 2


def test_length_histogram():
    h = length_histogram([50, 250, 1999, 5000])
    assert h["0-200"] == 1 and h["200-400"] == 1 and h["1800-2000"] == 1 and h[">2000"] == 1
