"""
Semantic faithfulness evaluation using Natural Language Inference (NLI).

Why NLI over another LLM call?
  Using a second LLM to judge the first LLM introduces a trust dependency —
  you're trusting the judge to be faithful, not just verifying the answer.
  NLI is mechanical: a classifier trained on human-annotated textual entailment
  pairs that answers "does this premise entail this hypothesis?" without
  generating any new text, so there's nothing for it to hallucinate.

Architecture:
  answer sentences
       ↓
  (sentence, chunk) pairs for each retrieved chunk
       ↓
  NLI cross-encoder → entailment / neutral / contradiction probabilities
       ↓
  per-sentence: max entailment across chunks, flag contradictions
       ↓
  faithfulness_score = fraction of sentences entailed by retrieved context

The NLI model (cross-encoder/nli-deberta-v3-small, ~180MB, fine-tuned on
MNLI + SNLI + FEVER) is loaded lazily and cached — only imported if this
module is actually used, same pattern as reranker.py.

FAITHFULNESS_BACKEND env var:
  "nli" (default) — real NLI via sentence-transformers (requires torch)
  "none"          — disabled; returns a null result (torch-free deploys)
"""
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache

FAITHFULNESS_BACKEND = os.environ.get("FAITHFULNESS_BACKEND", "nli").lower()
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
ENTAILMENT_THRESHOLD = 0.5   # min entailment probability to consider a sentence grounded
CONTRADICTION_THRESHOLD = 0.4 # flag if contradiction probability exceeds this


@dataclass
class SentenceFaithfulness:
    sentence: str
    max_entailment: float       # highest entailment prob across all retrieved chunks
    max_contradiction: float    # highest contradiction prob across all chunks
    grounded: bool              # entailment >= ENTAILMENT_THRESHOLD
    contradicted: bool          # contradiction >= CONTRADICTION_THRESHOLD


@dataclass
class FaithfulnessResult:
    faithfulness_score: float           # fraction of sentences that are grounded
    contradiction_rate: float           # fraction contradicted by retrieved context
    sentence_scores: list[SentenceFaithfulness] = field(default_factory=list)
    backend_used: str = "nli"
    n_sentences: int = 0
    n_grounded: int = 0
    n_contradicted: int = 0

    @property
    def is_faithful(self) -> bool:
        return self.faithfulness_score >= 0.8


@lru_cache(maxsize=1)
def _load_nli_model():
    from sentence_transformers import CrossEncoder
    # Label order for this model: CONTRADICTION=0, ENTAILMENT=1, NEUTRAL=2
    # (confirmed from model card — not all NLI models use the same label order)
    return CrossEncoder(NLI_MODEL)


def _split_sentences(text: str) -> list[str]:
    """Split answer into sentences, filtering out very short fragments that
    carry no factual claim (e.g. 'Yes.' or 'Here is a summary:')."""
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 20]


def score_faithfulness(answer: str, retrieved_chunks: list[str]) -> FaithfulnessResult:
    """Score whether the answer's sentences are entailed by the retrieved context.

    Args:
        answer: The LLM's generated answer text.
        retrieved_chunks: The text of the chunks passed to the LLM as context.

    Returns:
        FaithfulnessResult with per-sentence scores and an aggregate
        faithfulness_score in [0, 1].
    """
    if FAITHFULNESS_BACKEND == "none" or not retrieved_chunks:
        return FaithfulnessResult(
            faithfulness_score=0.0,
            contradiction_rate=0.0,
            backend_used="none",
        )

    sentences = _split_sentences(answer)
    if not sentences:
        # Empty or sub-20-char answer: gate it rather than treating as "fully faithful".
        # An answer with nothing to check should not pass a faithfulness gate.
        return FaithfulnessResult(
            faithfulness_score=0.0,
            contradiction_rate=0.0,
            backend_used="nli",
            n_sentences=0,
        )

    try:
        model = _load_nli_model()

        # Build all (sentence, chunk) pairs for batch inference —
        # more efficient than calling model.predict() per sentence.
        pairs = [
            (sentence, chunk)
            for sentence in sentences
            for chunk in retrieved_chunks
        ]
        n_chunks = len(retrieved_chunks)

        import numpy as np
        raw_scores = model.predict(pairs, apply_softmax=True)
        # Shape: (n_sentences * n_chunks, 3) — columns: [contradiction, entailment, neutral]
        scores = np.array(raw_scores).reshape(len(sentences), n_chunks, 3)

        sentence_results = []
        for i, sentence in enumerate(sentences):
            entailment_probs = scores[i, :, 1]    # entailment column
            contradiction_probs = scores[i, :, 0]  # contradiction column
            max_ent = float(entailment_probs.max())
            max_con = float(contradiction_probs.max())
            sentence_results.append(SentenceFaithfulness(
                sentence=sentence,
                max_entailment=round(max_ent, 3),
                max_contradiction=round(max_con, 3),
                grounded=max_ent >= ENTAILMENT_THRESHOLD,
                contradicted=max_con >= CONTRADICTION_THRESHOLD,
            ))

        n_grounded = sum(s.grounded for s in sentence_results)
        n_contradicted = sum(s.contradicted for s in sentence_results)
        n = len(sentences)

        return FaithfulnessResult(
            faithfulness_score=round(n_grounded / n, 3) if n else 0.0,
            contradiction_rate=round(n_contradicted / n, 3) if n else 0.0,
            sentence_scores=sentence_results,
            backend_used="nli",
            n_sentences=n,
            n_grounded=n_grounded,
            n_contradicted=n_contradicted,
        )

    except Exception as e:
        return FaithfulnessResult(
            faithfulness_score=0.0,
            contradiction_rate=0.0,
            backend_used=f"error:{e}",
        )
