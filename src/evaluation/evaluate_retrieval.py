"""
First-pass retrieval evaluation, run right after Phase 15 rather than deferred to
Phase 19 -- FEATURES.md is explicit that retrieval evaluation belongs early, since it's
the single biggest quality lever in a RAG system and problems here are cheapest to catch
before an agent is built on top of unreliable retrieval.

Scoring reports two things per question, not just pass/fail: whether the correct
document appears ANYWHERE in the top-K (hit rate @ k), and, when it does, at what
rank (used for Mean Reciprocal Rank -- MRR rewards ranking the right document 1st
over ranking it 5th, which a bare hit-rate number can't distinguish). Run against
both `semantic_search` (embedding-only) and `hybrid_search` (BM25 + semantic via
RRF), since the whole point of adding hybrid retrieval was to improve on
semantic-only -- a claim that needs this comparison to back it up, not just the
existence of the code path (see README's "Retrieval evaluation" section for the
real numbers this script produces).
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src.retrieval.search import SearchResult, hybrid_search, semantic_search


@dataclass
class RetrievalEvalResult:
    question: str
    expected_document_id: str
    correct_document_in_top_k: bool
    rank: int | None  # 1-indexed rank of the expected document, None if not found in top-K
    top_result_document_id: str | None
    top_result_section: str | None
    # Stricter than document-level: does the TOP result's own section match what's
    # expected? With only 4 candidate documents in this corpus, document-level hit
    # rate saturates at 100% almost immediately and stops being a useful signal --
    # this is the check that still discriminates between retrieval quality.
    top_result_section_correct: bool


def load_questions(path: str = "data/evaluation/retrieval_questions.json") -> list[dict]:
    return json.loads(Path(path).read_text())


def _rank_of(results: list[SearchResult], expected_document_id: str) -> int | None:
    for i, r in enumerate(results, start=1):
        if r.document_id == expected_document_id:
            return i
    return None


def evaluate_one(
    question_spec: dict, search_fn: Callable[..., list[SearchResult]], top_k: int = 5,
) -> RetrievalEvalResult:
    results = search_fn(question_spec["question"], top_k=top_k)
    rank = _rank_of(results, question_spec["expected_document_id"])
    top_result_section = results[0].section_title if results else None
    expected_section_fragment = question_spec.get("expected_section_contains")

    return RetrievalEvalResult(
        question=question_spec["question"],
        expected_document_id=question_spec["expected_document_id"],
        correct_document_in_top_k=rank is not None,
        rank=rank,
        top_result_document_id=results[0].document_id if results else None,
        top_result_section=top_result_section,
        top_result_section_correct=bool(
            expected_section_fragment and top_result_section
            and expected_section_fragment.lower() in top_result_section.lower()
        ),
    )


def _summarize(results: list[RetrievalEvalResult]) -> dict:
    passed = sum(1 for r in results if r.correct_document_in_top_k)
    section_passed = sum(1 for r in results if r.top_result_section_correct)
    # MRR: 1/rank for a hit, 0 for a miss, averaged across all questions -- a
    # document found at rank 1 every time gives MRR=1.0, found at rank 5 every
    # time gives MRR=0.2, distinguishing "always right" from "eventually right".
    mrr = sum((1.0 / r.rank) if r.rank else 0.0 for r in results) / len(results) if results else 0.0
    return {
        "total": len(results),
        "passed": passed,
        "hit_rate_pct": round((passed / len(results)) * 100, 1) if results else 0.0,
        "mrr": round(mrr, 3),
        "section_passed": section_passed,
        "section_accuracy_pct": round((section_passed / len(results)) * 100, 1) if results else 0.0,
        "results": results,
    }


def run_evaluation(path: str = "data/evaluation/retrieval_questions.json", top_k: int = 5) -> dict:
    """Runs the same question set through both semantic-only and hybrid search,
    so the comparison is apples-to-apples (same questions, same top_k, same run)."""
    questions = load_questions(path)
    return {
        "semantic": _summarize([evaluate_one(q, semantic_search, top_k) for q in questions]),
        "hybrid": _summarize([evaluate_one(q, hybrid_search, top_k) for q in questions]),
    }


if __name__ == "__main__":
    summary = run_evaluation()
    for method_name in ("semantic", "hybrid"):
        s = summary[method_name]
        print(f"\n=== {method_name} search ===")
        print(f"Hit rate @ K: {s['passed']}/{s['total']} ({s['hit_rate_pct']}%)   MRR: {s['mrr']}")
        print(f"Top-1 section accuracy: {s['section_passed']}/{s['total']} ({s['section_accuracy_pct']}%)")
        for r in s["results"]:
            status = "PASS" if r.correct_document_in_top_k else "FAIL"
            section_status = "PASS" if r.top_result_section_correct else "FAIL"
            print(f"[{status}] rank={r.rank} section=[{section_status}] {r.question}")
            print(f"       expected={r.expected_document_id} got={r.top_result_document_id} "
                  f"got_section={r.top_result_section}")
