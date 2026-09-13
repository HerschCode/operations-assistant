"""
Retrieval evaluation script for the Northstar Operations Assistant.

Loads data/evaluation/eval_dataset.json and evaluates three retrieval strategies
(semantic, hybrid, BM25) against every in-scope question that has an
expected_document_id, then computes:

  - Hit@1 / Hit@3 / Hit@5  (correct doc appears in top-1/3/5 results)
  - MRR (mean reciprocal rank; 1/rank at first match, 0 if not found in top-5)

Results are broken down by question category and saved to reports/p2_retrieval_eval.json.

Usage (from repo root):
    python scripts/evaluate_retrieval.py

Requires an indexed ChromaDB at data/chroma/ and the GROQ_API_KEY env var (or
a populated .env file) because semantic_search calls embed_query which may use
the configured embedding model.
"""
import json
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

load_dotenv()

from src.retrieval.search import (  # noqa: E402 — must follow sys.path insert
    bm25_search,
    hybrid_search,
    semantic_search,
)

TOP_K = 5
STRATEGIES = ["semantic", "hybrid", "bm25"]
EVAL_DATASET_PATH = REPO_ROOT / "data" / "evaluation" / "eval_dataset.json"
REPORT_PATH = REPO_ROOT / "reports" / "p2_retrieval_eval.json"


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _run_strategy(strategy: str, question: str) -> list:
    """Run one retrieval strategy and return the list of SearchResult objects."""
    if strategy == "semantic":
        return semantic_search(question, top_k=TOP_K)
    elif strategy == "hybrid":
        return hybrid_search(question, top_k=TOP_K)
    elif strategy == "bm25":
        return bm25_search(question, top_k=TOP_K)
    raise ValueError(f"Unknown strategy: {strategy}")


def _first_match_rank(results: list, expected_doc_id: str) -> Optional[int]:
    """Return 1-based rank of the first result whose document_id matches, or None."""
    for rank, result in enumerate(results, start=1):
        if result.document_id == expected_doc_id:
            return rank
    return None


def _compute_metrics(ranks: list[Optional[int]]) -> dict:
    """
    Given a list of first-match ranks (None = not found in top-5), compute:
      hit@1, hit@3, hit@5  as percentages (0-100)
      mrr                  as a float in [0, 1]
    """
    total = len(ranks)
    if total == 0:
        return {"hit_at_1": 0.0, "hit_at_3": 0.0, "hit_at_5": 0.0, "mrr": 0.0, "n": 0}

    hit1 = sum(1 for r in ranks if r is not None and r <= 1)
    hit3 = sum(1 for r in ranks if r is not None and r <= 3)
    hit5 = sum(1 for r in ranks if r is not None and r <= 5)
    mrr = sum((1.0 / r) if r is not None else 0.0 for r in ranks) / total

    return {
        "hit_at_1": round(hit1 / total * 100, 1),
        "hit_at_3": round(hit3 / total * 100, 1),
        "hit_at_5": round(hit5 / total * 100, 1),
        "mrr": round(mrr, 4),
        "n": total,
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(dataset: list[dict]) -> dict:
    """
    Evaluate all in-scope questions with an expected_document_id.
    Returns a nested dict: {strategy -> {category -> [ranks]}} for later metric
    computation, plus per-question records.
    """
    in_scope = [
        q for q in dataset
        if q.get("in_scope") and q.get("expected_document_id")
    ]

    print(f"Evaluating {len(in_scope)} in-scope questions (top_k={TOP_K}) ...")
    print()

    # {strategy -> list of {question_id, category, rank}}
    per_strategy_records: dict[str, list[dict]] = {s: [] for s in STRATEGIES}

    for i, question_entry in enumerate(in_scope, start=1):
        qid = question_entry["id"]
        question = question_entry["question"]
        expected_doc = question_entry["expected_document_id"]
        category = question_entry.get("category", "unknown")

        for strategy in STRATEGIES:
            t0 = time.perf_counter()
            try:
                results = _run_strategy(strategy, question)
            except Exception as exc:
                print(f"  [WARN] q{qid} {strategy}: {exc}")
                results = []
            latency_ms = (time.perf_counter() - t0) * 1000

            rank = _first_match_rank(results, expected_doc)
            per_strategy_records[strategy].append({
                "question_id": qid,
                "category": category,
                "question": question,
                "expected_document_id": expected_doc,
                "rank": rank,
                "latency_ms": round(latency_ms, 1),
            })

        if i % 10 == 0 or i == len(in_scope):
            print(f"  ... {i}/{len(in_scope)} questions evaluated")

    return per_strategy_records


def aggregate_metrics(per_strategy_records: dict[str, list[dict]]) -> dict:
    """
    Compute overall and per-category Hit@k and MRR for each strategy.
    Returns {strategy -> {overall: {...}, by_category: {cat: {...}}}}.
    """
    result: dict[str, dict] = {}

    for strategy, records in per_strategy_records.items():
        all_ranks = [r["rank"] for r in records]
        overall_metrics = _compute_metrics(all_ranks)

        # group by category
        categories: dict[str, list] = {}
        for rec in records:
            cat = rec["category"]
            categories.setdefault(cat, []).append(rec["rank"])

        by_category = {
            cat: _compute_metrics(ranks)
            for cat, ranks in sorted(categories.items())
        }

        result[strategy] = {
            "overall": overall_metrics,
            "by_category": by_category,
        }

    return result


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_summary(metrics: dict) -> None:
    """Print a compact summary table comparing the three strategies."""
    col_w = 12

    header_parts = [f"{'Metric':<20}"]
    for strategy in STRATEGIES:
        header_parts.append(f"{strategy:>{col_w}}")
    print("".join(header_parts))
    print("-" * (20 + col_w * len(STRATEGIES)))

    metric_keys = [
        ("Hit@1 (%)", "hit_at_1"),
        ("Hit@3 (%)", "hit_at_3"),
        ("Hit@5 (%)", "hit_at_5"),
        ("MRR",       "mrr"),
        ("N",         "n"),
    ]
    for label, key in metric_keys:
        row = [f"{label:<20}"]
        for strategy in STRATEGIES:
            val = metrics[strategy]["overall"].get(key, "-")
            if isinstance(val, float):
                row.append(f"{val:>{col_w}.4f}" if key == "mrr" else f"{val:>{col_w}.1f}")
            else:
                row.append(f"{val:>{col_w}}")
        print("".join(row))

    print()
    print("By category (Hit@5 %):")
    categories = sorted(
        set(cat for s in metrics.values() for cat in s["by_category"])
    )
    cat_header = [f"{'Category':<26}"]
    for s in STRATEGIES:
        cat_header.append(f"{s:>{col_w}}")
    print("".join(cat_header))
    print("-" * (26 + col_w * len(STRATEGIES)))

    for cat in categories:
        row = [f"{cat:<26}"]
        for strategy in STRATEGIES:
            cat_data = metrics[strategy]["by_category"].get(cat)
            if cat_data:
                row.append(f"{cat_data['hit_at_5']:>{col_w}.1f}")
            else:
                row.append(f"{'—':>{col_w}}")
        print("".join(row))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not EVAL_DATASET_PATH.exists():
        print(f"Eval dataset not found: {EVAL_DATASET_PATH}")
        sys.exit(1)

    dataset = json.loads(EVAL_DATASET_PATH.read_text(encoding="utf-8"))

    per_strategy_records = run_evaluation(dataset)
    metrics = aggregate_metrics(per_strategy_records)

    print()
    print("=" * 60)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 60)
    print_summary(metrics)

    report = {
        "eval_dataset": str(EVAL_DATASET_PATH),
        "top_k": TOP_K,
        "strategies": STRATEGIES,
        "metrics": metrics,
        "records": per_strategy_records,
    }

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print()
    print(f"Saved → {REPORT_PATH}")


if __name__ == "__main__":
    main()
