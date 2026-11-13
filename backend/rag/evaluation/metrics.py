"""Retrieval metrics: Precision@k, Recall@k, Hit@k, MRR, nDCG@k.

Ground truth khai báo theo (file, trang), KHÔNG theo chunk_id, để dùng chung
một bộ câu hỏi cho mọi chiến lược chunking (mỗi chiến lược có chunk_id khác nhau).

Định dạng cột ground_truth trong questions.csv:
    "paper_a.pdf:3-4; paper_b.pdf:7,9"
Mỗi phần ngăn bởi ";" là một "bằng chứng" (evidence). Không ghi trang => cả file.

Định nghĩa:
- Một chunk RELEVANT nếu cùng file và khoảng trang giao với một evidence.
- Precision@k = số chunk relevant trong top-k / k
- Recall@k    = số evidence được ít nhất một chunk top-k bao phủ / tổng số evidence
                (câu hỏi tổng hợp 2 nguồn mà chỉ tìm được 1 => Recall = 0.5)
- Hit@k       = 1 nếu có ít nhất 1 chunk relevant trong top-k
- MRR         = 1 / hạng của chunk relevant đầu tiên (trên toàn danh sách trả về)
- nDCG@k      = DCG nhị phân / DCG lý tưởng (lý tưởng = min(k, số chunk relevant trong index))
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from rag.schemas import Chunk


@dataclass
class Evidence:
    source: str
    pages: set[int] = field(default_factory=set)   # rỗng = cả tài liệu

    def matches(self, chunk: Chunk) -> bool:
        if _norm_source(chunk.source) != _norm_source(self.source):
            return False
        return not self.pages or any(p in self.pages for p in chunk.pages)


def _norm_source(name: str) -> str:
    # NFC: tên file tiếng Việt trên đĩa có thể ở dạng NFD, gõ tay trong CSV là NFC
    return unicodedata.normalize("NFC", Path(name.strip()).stem).lower()


def _parse_pages(spec: str) -> set[int]:
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = (int(x) for x in part.split("-", 1))
            pages.update(range(min(a, b), max(a, b) + 1))
        else:
            pages.add(int(part))
    return pages


def parse_ground_truth(spec: str) -> list[Evidence]:
    evidences: list[Evidence] = []
    for item in str(spec or "").split(";"):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            source, pages = item.rsplit(":", 1)
            evidences.append(Evidence(source.strip(), _parse_pages(pages)))
        else:
            evidences.append(Evidence(item))
    return evidences


def is_relevant(chunk: Chunk, evidences: list[Evidence]) -> bool:
    return any(ev.matches(chunk) for ev in evidences)


def precision_at_k(rels: list[bool], k: int) -> float:
    return sum(rels[:k]) / k if k else 0.0


def recall_at_k(chunks: list[Chunk], evidences: list[Evidence], k: int) -> float:
    if not evidences:
        return 0.0
    top = chunks[:k]
    return sum(any(ev.matches(c) for c in top) for ev in evidences) / len(evidences)


def hit_at_k(rels: list[bool], k: int) -> float:
    return 1.0 if any(rels[:k]) else 0.0


def mrr(rels: list[bool]) -> float:
    for i, r in enumerate(rels, start=1):
        if r:
            return 1.0 / i
    return 0.0


def ndcg_at_k(rels: list[bool], k: int, n_relevant_total: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(rels[:k], start=1) if r)
    ideal_hits = min(k, n_relevant_total)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def evaluate_ranking(chunks: list[Chunk], evidences: list[Evidence], k: int, n_relevant_total: int) -> dict[str, float]:
    rels = [is_relevant(c, evidences) for c in chunks]
    return {
        f"precision@{k}": precision_at_k(rels, k),
        f"recall@{k}": recall_at_k(chunks, evidences, k),
        f"hit@{k}": hit_at_k(rels, k),
        "mrr": mrr(rels),
        f"ndcg@{k}": ndcg_at_k(rels, k, n_relevant_total),
    }
