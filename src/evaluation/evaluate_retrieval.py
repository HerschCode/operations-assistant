"""
First-pass retrieval evaluation, run right after Phase 15 rather than deferred to
Phase 19 -- FEATURES.md is explicit that retrieval evaluation belongs early, since it's
the single biggest quality lever in a RAG system and problems here are cheapest to catch
before an agent is built on top of unreliable retrieval.

Scoring is deliberately simple: for each question, does the correct document appear
ANYWHERE in the top-K results, and does the top-1 result's section match what's
expected. This is a coarse pass/fail, not similarity-score calibration -- good enough
to catch "retrieval is fundamentally broken for this question" before Phase 19 adds
more rigorous scoring (citation correctness, groundedness) on top of a working base.
"""
import json
from dataclasses import dataclass
from pathlib import Path

from src.retrieval.search import semantic_search


@dataclass
class RetrievalEvalResult:
    question: str
    expected_document_id: str
    correct_document_in_top_k: bool
    top_result_document_id: str | None
    top_result_section: str | None


def load_questions(path: str = "data/evaluation/retrieval_questions.json") -> list[dict]:
    return json.loads(Path(path).read_text())


def evaluate_one(question_spec: dict, top_k: int = 5) -> RetrievalEvalResult:
    results = semantic_search(question_spec["question"], top_k=top_k)
    result_doc_ids = [r.document_id for r in results]

    return RetrievalEvalResult(
        question=question_spec["question"],
        expected_document_id=question_spec["expected_document_id"],
        correct_document_in_top_k=question_spec["expected_document_id"] in result_doc_ids,
        top_result_document_id=results[0].document_id if results else None,
        top_result_section=results[0].section_title if results else None,
    )


def run_evaluation(path: str = "data/evaluation/retrieval_questions.json") -> dict:
    questions = load_questions(path)
    results = [evaluate_one(q) for q in questions]

    passed = sum(1 for r in results if r.correct_document_in_top_k)
    return {
        "total": len(results),
        "passed": passed,
        "pass_rate_pct": round((passed / len(results)) * 100, 1) if results else 0.0,
        "results": results,
    }


if __name__ == "__main__":
    summary = run_evaluation()
    print(f"Retrieval eval: {summary['passed']}/{summary['total']} "
          f"({summary['pass_rate_pct']}%) correct document in top-K\n")
    for r in summary["results"]:
        status = "PASS" if r.correct_document_in_top_k else "FAIL"
        print(f"[{status}] {r.question}")
        print(f"       expected={r.expected_document_id} got={r.top_result_document_id}")
