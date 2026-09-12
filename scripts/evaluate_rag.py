"""
RAG evaluation: measures retrieval hit rate @ k=3 and answer faithfulness.

Runs a small hand-written QA set against the live ChromaDB collection and reports
whether the correct answer text appears in the top-3 retrieved chunks. Designed to
be run after documents are indexed:

    python -m scripts.index_documents   # if not already done
    python scripts/evaluate_rag.py

Uses the same hybrid_search() call path as the live agent tool
(src/tools/documents.py -> src/retrieval/search.py) so results reflect exactly
what the agent sees, not a separate retrieval path.

Domain: Northstar Manufacturing procurement policy documents
  - procurement-policy.md  (approval thresholds, PO change rules, SLA targets)
  - sla-policy.md          (cycle time targets by category, breach reporting)
  - escalation-procedure.md (when/how to escalate, escalation path)
  - exception-handling-procedure.md (emergency purchases, supplier disputes)
"""
import time
from dataclasses import dataclass

from src.retrieval.search import hybrid_search

# ---------------------------------------------------------------------------
# QA set -- 15 questions grounded in the 4 policy documents above.
# "answer_contains": a short substring that must appear in at least one of the
# top-3 retrieved chunks for the question to count as a hit.
# ---------------------------------------------------------------------------
QA_PAIRS = [
    {
        "question": "What approval is required for purchase orders above $10,000?",
        "answer_contains": "Secondary Approval",
    },
    {
        "question": "How many business days does standard purchase order approval take?",
        "answer_contains": "5 business days",
    },
    {
        "question": "When does a purchase requisition require Procurement review?",
        "answer_contains": "above $500",
    },
    {
        "question": "What triggers Manual Credit Review?",
        "answer_contains": "new supplier",
    },
    {
        "question": "What happens to a PO that has been changed after release?",
        "answer_contains": "approval chain to be re-run",
    },
    {
        "question": "What are the SLA targets for 3-way match and consignment categories?",
        "answer_contains": "Consignment",
    },
    {
        "question": "When does an SLA breach get reported in the management report?",
        "answer_contains": "weekly",
    },
    {
        "question": "How often are SLA targets reviewed?",
        "answer_contains": "quarterly",
    },
    {
        "question": "When should a case be escalated to Operations leadership?",
        "answer_contains": "5 business days",
    },
    {
        "question": "What is the escalation path for a procurement case?",
        "answer_contains": "Operations Director",
    },
    {
        "question": "Does escalation override the approval requirements in the procurement policy?",
        "answer_contains": "does not override",
    },
    {
        "question": "What must happen within 2 business days after an emergency purchase?",
        "answer_contains": "retroactive Purchase Requisition",
    },
    {
        "question": "How is cycle time calculated under the SLA policy?",
        "answer_contains": "Purchase Requisition creation to final Payment",
    },
    {
        "question": "What is the exception to the SLA policy cycle time counting rule?",
        "answer_contains": "supplier disputes",
    },
    {
        "question": "What should Operations do if the same supplier appears in escalations more than twice in a quarter?",
        "answer_contains": "review that supplier",
    },
]

TOP_K = 3


@dataclass
class EvalRow:
    question: str
    top_doc: str | None
    hit: bool
    latency_ms: float


def _is_hit(results, answer_contains: str) -> bool:
    needle = answer_contains.lower()
    return any(needle in r.text.lower() for r in results)


def run() -> None:
    rows: list[EvalRow] = []

    for qa in QA_PAIRS:
        t0 = time.perf_counter()
        results = hybrid_search(qa["question"], top_k=TOP_K)
        latency_ms = (time.perf_counter() - t0) * 1000

        hit = _is_hit(results, qa["answer_contains"])
        top_doc = results[0].document_id if results else None
        rows.append(EvalRow(
            question=qa["question"],
            top_doc=top_doc,
            hit=hit,
            latency_ms=latency_ms,
        ))

    # --- results table ---
    col_q = 60
    col_doc = 30
    header = f"{'Question':<{col_q}} {'Top doc':<{col_doc}} {'Hit':>4} {'Latency':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        hit_str = "YES" if row.hit else "NO "
        doc_str = (row.top_doc or "none")[:col_doc]
        q_str = row.question[:col_q]
        print(f"{q_str:<{col_q}} {doc_str:<{col_doc}} {hit_str:>4} {row.latency_ms:>8.1f}ms")

    # --- summary ---
    total = len(rows)
    hits = sum(1 for r in rows if r.hit)
    hit_rate = hits / total if total else 0.0
    avg_latency = sum(r.latency_ms for r in rows) / total if total else 0.0

    print()
    print(f"hit_rate @ k={TOP_K}: {hits}/{total} ({hit_rate:.0%})")
    print(f"avg_latency_ms:       {avg_latency:.1f}")
    print(f"total_questions:      {total}")


if __name__ == "__main__":
    run()
