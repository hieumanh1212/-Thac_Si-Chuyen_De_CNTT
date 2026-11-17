import pytest

from rag.evaluation.metrics import evaluate_ranking, parse_ground_truth
from rag.generation.context_builder import citation_label, render_citations
from rag.schemas import Chunk


def chunk(cid, source, page, end_page=None, section="", **meta):
    return Chunk(chunk_id=cid, text="x", source=source, page=page, end_page=end_page or page,
                 section=section, strategy="t", metadata=meta)


def test_parse_ground_truth():
    evs = parse_ground_truth("a.pdf:3-4,7; b.pdf")
    assert evs[0].source == "a.pdf" and evs[0].pages == {3, 4, 7}
    assert evs[1].source == "b.pdf" and evs[1].pages == set()


def test_metrics_multi_source():
    evs = parse_ground_truth("a.pdf:3; b.pdf:7")
    ranked = [chunk(0, "c.pdf", 1), chunk(1, "a.pdf", 2, 3), chunk(2, "a.pdf", 3), chunk(3, "b.pdf", 7)]
    m = evaluate_ranking(ranked, evs, k=3, n_relevant_total=3)
    assert m["precision@3"] == pytest.approx(2 / 3)
    assert m["recall@3"] == pytest.approx(0.5)      # chỉ a.pdf được bao phủ trong top-3
    assert m["hit@3"] == 1.0
    assert m["mrr"] == pytest.approx(0.5)
    assert 0 < m["ndcg@3"] < 1


def test_source_match_ignores_extension_and_case():
    evs = parse_ground_truth("Paper_A:5")
    m = evaluate_ranking([chunk(0, "paper_a.pdf", 5)], evs, k=3, n_relevant_total=1)
    assert m["hit@3"] == 1.0 and m["ndcg@3"] == pytest.approx(1.0)


def test_source_match_ignores_unicode_form():
    import unicodedata
    nfd_name = unicodedata.normalize("NFD", "Báo cáo Big Data.pdf")
    evs = parse_ground_truth("Báo cáo Big Data.pdf:3")
    m = evaluate_ranking([chunk(0, nfd_name, 3)], evs, k=3, n_relevant_total=1)
    assert m["hit@3"] == 1.0


def test_citation_label_and_render():
    c = chunk(0, "a.pdf", 8, section="4.2 Results", short_ref="Nguyen et al., 2024")
    label = citation_label(c, "{short_ref}, {section}, trang {pages}")
    assert label == "Nguyen et al., 2024, 4.2 Results, trang 8"
    no_section = citation_label(chunk(1, "b.pdf", 2, 3, title="B"), "{short_ref}, {section}, trang {pages}")
    assert no_section == "B, trang 2-3"
    rendered = render_citations("Đạt 91,2% [1]. Nhanh hơn [1, 2]. Năm [2017].", [label, no_section])
    assert rendered == (f"Đạt 91,2% [{label}]. Nhanh hơn [{label}][{no_section}]. Năm [2017].")


# ---------------- gọi LLM khi đánh giá: thử lại khi vượt giới hạn (429) ----------------
import pytest  # noqa: E402

from rag.evaluation.run_eval import DailyQuotaExceeded, ask_with_retry  # noqa: E402


class _RateLimited(Exception):
    status_code = 429


class _FakePipe:
    def __init__(self, fail_times: int, exc: Exception) -> None:
        self.calls, self.fail_times, self.exc = 0, fail_times, exc

    def ask(self, *args, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc
        return "ok"


def test_ask_with_retry_recovers_from_rate_limit():
    pipe = _FakePipe(2, _RateLimited("Error code: 429"))
    assert ask_with_retry(pipe, "q", "semantic", 5, waits=(0, 0, 0)) == "ok"
    assert pipe.calls == 3


def test_ask_with_retry_gives_up_and_does_not_retry_other_errors():
    with pytest.raises(_RateLimited):
        ask_with_retry(_FakePipe(9, _RateLimited("429")), "q", "semantic", 5, waits=(0, 0))
    other = _FakePipe(1, ValueError("bad request"))
    with pytest.raises(ValueError):
        ask_with_retry(other, "q", "semantic", 5, waits=(0, 0))
    assert other.calls == 1


def test_ask_with_retry_stops_immediately_on_daily_quota():
    exc = _RateLimited("Error code: 429")
    exc.body = [{"error": {"message": "Generate content requests per day limit exceeded on Free Tier"}}]
    pipe = _FakePipe(9, exc)
    with pytest.raises(DailyQuotaExceeded):
        ask_with_retry(pipe, "q", "semantic", 5, waits=(0, 0, 0))
    assert pipe.calls == 1
