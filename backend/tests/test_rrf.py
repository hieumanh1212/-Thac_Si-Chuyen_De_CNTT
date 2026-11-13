import pytest

from rag.retrieval.rrf import reciprocal_rank_fusion


def test_rrf_rewards_items_in_both_lists():
    fused = reciprocal_rank_fusion({"bm25": [1, 2, 3], "faiss": [3, 4, 1]}, k=60)
    ids = [cid for cid, _, _ in fused]
    # 1 và 3 có mặt ở cả hai danh sách (1/61 + 1/63) => đứng đầu
    assert set(ids[:2]) == {1, 3}
    # 2 và 4 chỉ có ở một danh sách (1/62) => xếp sau
    assert set(ids[2:]) == {2, 4}


def test_rrf_scores_and_ranks():
    fused = dict((cid, (score, ranks)) for cid, score, ranks in
                 reciprocal_rank_fusion({"bm25": [10], "faiss": [20, 10]}, k=60))
    assert fused[10][0] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[10][1] == {"bm25": 1, "faiss": 2}
    assert fused[20][0] == pytest.approx(1 / 61)


def test_rrf_ignores_duplicates_within_list():
    fused = reciprocal_rank_fusion({"bm25": [5, 5, 6]}, k=60)
    scores = {cid: s for cid, s, _ in fused}
    assert scores[5] == pytest.approx(1 / 61)
    assert scores[6] == pytest.approx(1 / 62)
