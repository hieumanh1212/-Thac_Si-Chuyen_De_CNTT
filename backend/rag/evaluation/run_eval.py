"""Chạy đánh giá: (chiến lược chunking) x (bm25 | faiss | hybrid) x (bộ câu hỏi).

Chạy:
    python -m rag.evaluation.run_eval                     # chỉ retrieval metrics
    python -m rag.evaluation.run_eval --with-llm          # thêm câu trả lời LLM (mode hybrid)

Kết quả: eval/results/eval_<thời gian>.xlsx gồm các sheet
    per_query        - metrics từng câu x chiến lược x mode
    summary          - trung bình theo chiến lược x mode
    summary_by_type  - trung bình theo loại câu hỏi (keyword/semantic/synthesis)
    answers          - câu trả lời LLM + cột trống để chấm tay (đúng/đủ/bám nguồn/citation)
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from typing import Any

import pandas as pd

from rag.config import load_config, resolve_path, setup_logging
from rag.evaluation.metrics import evaluate_ranking, is_relevant, parse_ground_truth
from rag.generation.context_builder import format_pages
from rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"id", "query", "type", "ground_truth"}


def load_questions(cfg: dict[str, Any]) -> pd.DataFrame:
    path = resolve_path(cfg["paths"]["eval_questions"])
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)  # utf-8-sig: đọc được CSV lưu từ Excel
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} thiếu cột: {missing}")
    return df


def evaluate(cfg: dict[str, Any], strategies: list[str] | None = None, modes: list[str] | None = None,
             with_llm: bool = False, pipeline: RAGPipeline | None = None) -> dict[str, pd.DataFrame]:
    pipe = pipeline or RAGPipeline(cfg)
    ecfg = cfg.get("evaluation", {})
    k = ecfg.get("k", 3)
    depth = max(ecfg.get("retrieve_depth", 10), k)
    modes = modes or ecfg.get("modes", ["bm25", "faiss", "hybrid"])
    strategies = strategies or pipe.available_strategies()
    if not strategies:
        raise RuntimeError("Chưa có index nào. Chạy: python -m rag.indexing.build_index")
    questions = load_questions(cfg)

    rows, answer_rows = [], []
    for strategy in strategies:
        retriever = pipe.retriever(strategy)
        all_chunks = list(retriever.chunks.values())
        for q in questions.itertuples(index=False):
            evidences = parse_ground_truth(q.ground_truth)
            n_rel = sum(is_relevant(c, evidences) for c in all_chunks)
            for mode in modes:
                sr = retriever.search(q.query, mode=mode, top_k=depth)
                ranked = [r.chunk for r in sr.results]
                metrics = evaluate_ranking(ranked, evidences, k, n_rel)
                rows.append({
                    "id": q.id, "type": q.type, "query": q.query, "strategy": strategy, "mode": mode,
                    **metrics,
                    "latency_ms": round(sr.timings_ms.get("total", 0.0), 1),
                    "n_relevant_in_index": n_rel,
                    f"top{k}": " | ".join(f"{c.source}:{format_pages(c)}" for c in ranked[:k]),
                    "ground_truth": q.ground_truth,
                })

            if with_llm:
                try:
                    ans = pipe.ask(q.query, strategy=strategy, mode="hybrid", top_k=cfg["retrieval"].get("top_k", 5))
                    answer, rendered, tokens = ans.answer, ans.answer_rendered, ans.context_tokens
                    llm_ms = round(ans.timings_ms.get("llm", 0.0), 1)
                except Exception as exc:  # lỗi API không làm hỏng cả lượt đánh giá
                    logger.error("LLM lỗi ở %s/%s: %s", strategy, q.id, exc)
                    answer, rendered, tokens, llm_ms = f"LỖI: {exc}", "", 0, 0.0
                answer_rows.append({
                    "id": q.id, "type": q.type, "query": q.query, "strategy": strategy,
                    "answer": rendered or answer, "context_tokens": tokens, "llm_ms": llm_ms,
                    "ground_truth": q.ground_truth,
                    # Chấm tay theo thang 0-2 (hoặc 0/1)
                    "correctness": "", "completeness": "", "faithfulness": "", "citation_correct": "", "notes": "",
                })
        logger.info("Xong chiến lược %s", strategy)

    per_query = pd.DataFrame(rows)
    metric_cols = [c for c in per_query.columns if "@" in c or c in ("mrr", "latency_ms")]
    summary = per_query.groupby(["strategy", "mode"])[metric_cols].mean().round(3).reset_index()
    by_type = per_query.groupby(["strategy", "mode", "type"])[metric_cols].mean().round(3).reset_index()
    out = {"questions": questions, "per_query": per_query, "summary": summary, "summary_by_type": by_type}
    if answer_rows:
        out["answers"] = pd.DataFrame(answer_rows)
    return out


def save_results(cfg: dict[str, Any], tables: dict[str, pd.DataFrame]) -> str:
    out_dir = resolve_path(cfg["paths"]["eval_results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"eval_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in tables.items():
            df.to_excel(writer, sheet_name=name, index=False)
    tables["summary"].to_csv(path.with_suffix(".summary.csv"), index=False, encoding="utf-8-sig")
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá retrieval + RAG")
    parser.add_argument("--strategies", nargs="*", default=None, help="Mặc định: mọi chiến lược đã có index")
    parser.add_argument("--modes", nargs="*", default=None, help="bm25 faiss hybrid")
    parser.add_argument("--with-llm", action="store_true", help="Sinh câu trả lời bằng LLM (tốn API)")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):          # console Windows (cp1252) không in được tiếng Việt
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    setup_logging()
    cfg = load_config(args.config)
    tables = evaluate(cfg, args.strategies, args.modes, args.with_llm)
    path = save_results(cfg, tables)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(tables["summary"].to_string(index=False))
    print(f"\nĐã lưu: {path}")


if __name__ == "__main__":
    main()
