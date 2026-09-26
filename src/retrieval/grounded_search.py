"""
Grounded RAG: retrieval → LLM → faithfulness gate.

The faithfulness evaluation (scripts/evaluate_faithfulness.py) found that the
paraphrase category scores 0% faithfulness — the LLM generates answers that
drift to outside knowledge when retrieved chunks are semantically adjacent but
not literal matches for the query. This module closes that gap by scoring the
LLM's answer against retrieved chunks via NLI and refusing to return an answer
that contradicts or ignores the retrieved context.

The gate is enforced at FAITHFULNESS_GATE_THRESHOLD (default 0.05, calibrated
2026-09-24 via scripts/calibrate_gate.py against a 67-question labeled set;
see reports/gate-calibration.md). When the answer's faithfulness score falls
below this threshold, the system returns a structured "insufficient information"
response rather than an unfaithful answer. This trades recall (sometimes saying
"I don't know" when a correct answer exists) for precision (never returning
answers that contradict the source documents).

GROUNDED_GATE_ENABLED env var:
  "true" (default) — enforce the faithfulness gate
  "false"          — run as plain RAG (no gate), useful for comparison runs

FAITHFULNESS_GATE_THRESHOLD env var:
  float in [0, 1], default 0.05. A sentence is grounded if its NLI entailment
  probability against any retrieved chunk exceeds this. The answer is gated if
  the fraction of grounded sentences < this threshold.
"""
import os
from dataclasses import dataclass, field

GROUNDED_GATE_ENABLED = os.environ.get("GROUNDED_GATE_ENABLED", "true").lower() in ("true", "1", "yes")
FAITHFULNESS_GATE_THRESHOLD = float(os.environ.get("FAITHFULNESS_GATE_THRESHOLD", "0.05"))
# GATE_METHOD: "support" (default since 2026-09-26) = lexical claim-support check (src/evaluation/claim_support.py);
# "nli" = the original NLI gate. On a labeled set (reports/gate_labeled_eval.json) the NLI gate passed correct,
# wrong-fact and off-context answers at the SAME rate (18.8% each on held-out questions) -- it did not discriminate.
GATE_METHOD = os.environ.get("GATE_METHOD", "support").lower()
SUPPORT_MIN_RECALL = float(os.environ.get("SUPPORT_MIN_RECALL", "0.65"))   # chosen on odd question ids only  # calibrated 2026-09-24; see reports/gate-calibration.md
INSUFFICIENT_DATA_MSG = (
    "The retrieved documents do not contain sufficient information to answer "
    "this question faithfully. Please consult the relevant policy documents directly "
    "or contact your operations team."
)


@dataclass
class GroundedAnswer:
    answer: str
    faithfulness_score: float
    gated: bool                          # True if the gate replaced the answer with fallback
    n_sentences: int = 0
    n_grounded: int = 0
    retrieved_chunks: list[str] = field(default_factory=list)
    backend_used: str = "nli"


def grounded_answer(
    question: str,
    llm_fn,
    top_k: int = 5,
    gate_threshold: float | None = None,
) -> GroundedAnswer:
    """Retrieve chunks, generate an LLM answer, and gate on faithfulness.

    Args:
        question: The user question.
        llm_fn: Callable(question, chunk_texts) -> str. The LLM generation
            function. Injected so this module has no hard dependency on a
            specific LLM provider.
        top_k: Number of chunks to retrieve.
        gate_threshold: Override FAITHFULNESS_GATE_THRESHOLD for this call.

    Returns:
        GroundedAnswer with the final answer (or fallback) and faithfulness scores.
    """
    from src.retrieval.search import reranked_search
    from src.evaluation.faithfulness import score_faithfulness
    from src.observability.spans import retrieve_span, gate_span

    threshold = gate_threshold if gate_threshold is not None else FAITHFULNESS_GATE_THRESHOLD

    with retrieve_span(question, top_k=top_k) as rspan:
        chunks = reranked_search(question, top_k=top_k)
        chunk_texts = [r.text for r in chunks]
        if rspan:
            rspan.set_attribute("n_chunks_returned", len(chunks))

    raw_answer = llm_fn(question, chunk_texts)

    if GATE_METHOD == "support":
        from src.evaluation.claim_support import answer_supported, split_sentences

        ok, detail = answer_supported(raw_answer, chunk_texts, SUPPORT_MIN_RECALL)
        n_ok = sum(d["numbers_supported"] and d["key_terms_supported"] and d["content_recall"] >= SUPPORT_MIN_RECALL
                   for d in detail)
        from src.evaluation.faithfulness import FaithfulnessResult

        result = FaithfulnessResult(faithfulness_score=round(n_ok / len(detail), 3) if detail else 0.0,
                                    contradiction_rate=0.0, backend_used="support",
                                    n_sentences=len(detail), n_grounded=n_ok)
        gated = GROUNDED_GATE_ENABLED and not ok
    else:
        result = score_faithfulness(raw_answer, chunk_texts)
        gated = GROUNDED_GATE_ENABLED and result.faithfulness_score < threshold
    final_answer = INSUFFICIENT_DATA_MSG if gated else raw_answer

    with gate_span(result.faithfulness_score, gated, threshold):
        pass

    return GroundedAnswer(
        answer=final_answer,
        faithfulness_score=result.faithfulness_score,
        gated=gated,
        n_sentences=result.n_sentences,
        n_grounded=result.n_grounded,
        retrieved_chunks=chunk_texts,
        backend_used=result.backend_used,
    )
