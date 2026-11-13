"""Reciprocal Rank Fusion.

    RRF(d) = Σ_r  w_r / (k + rank_r(d))

- Chỉ dùng THỨ HẠNG, không dùng điểm gốc => không cần chuẩn hóa điểm BM25 (0..∞)
  và cosine (-1..1) về cùng thang.
- k = 60 (giá trị gốc của Cormack et al., 2009) làm "phẳng" chênh lệch giữa hạng 1 và 2.
- Chunk xuất hiện ở cả hai danh sách được cộng điểm hai lần => được ưu tiên.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, Sequence[int]],
    k: int = 60,
    weights: Mapping[str, float] | None = None,
) -> list[tuple[int, float, dict[str, int]]]:
    """Trả về [(chunk_id, rrf_score, {tên_retriever: hạng}), ...] giảm dần theo điểm."""
    scores: dict[int, float] = defaultdict(float)
    ranks: dict[int, dict[str, int]] = defaultdict(dict)
    for name, ids in ranked_lists.items():
        w = weights.get(name, 1.0) if weights else 1.0
        rank = 0
        for cid in ids:
            if name in ranks[cid]:           # bỏ trùng trong cùng một danh sách
                continue
            rank += 1
            scores[cid] += w / (k + rank)
            ranks[cid][name] = rank
    return sorted(
        ((cid, score, ranks[cid]) for cid, score in scores.items()),
        key=lambda x: (-x[1], min(x[2].values())),   # hòa điểm => ưu tiên hạng tốt nhất
    )
