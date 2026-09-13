"""
Cross-encoder reranker for the RAG pipeline.

Architecture context:
  hybrid retrieval (BM25 + semantic, top-k=20 candidates)
      ↓
  cross-encoder reranker (scores each query-chunk pair jointly)
      ↓
  top-k=5 context passed to LLM

Why a cross-encoder on top of hybrid retrieval? The bi-encoder (all-MiniLM-L6-v2)
embeds query and document independently and compares them in a shared embedding space
-- fast, but it loses the interaction between query tokens and document tokens that
would let it distinguish "approval required for $5,000 POs" from "approval required
for $50,000 POs" when the query is about $50,000. A cross-encoder reads the full
(query, chunk) pair jointly and produces a single relevance score -- slower but
substantially more precise on fine-grained distinctions. The pipeline uses bi-encoder
retrieval for recall (fast, catch all plausibly relevant chunks) and cross-encoder
reranking for precision (pick the right one out of the candidates).

Deployment strategy (same pattern as P3's sentence_transformer embedding backend):
  RERANKER_BACKEND=cross_encoder (default): real cross-encoder via sentence-transformers
  RERANKER_BACKEND=none: reranking disabled (preserves original hybrid ranking)

The 'none' fallback exists because cross-encoder adds ~torch to the dependency tree,
which OOM-crashes Render's 512MB free tier (same issue that forced the switch from
sentence-transformers to ONNX for bi-encoder embeddings -- see src/retrieval/embeddings.py).
Set RERANKER_BACKEND=none in render.yaml (as EMBEDDING_BACKEND=tfidf is set in P3).
Locally: no env var needed, cross_encoder is the default and the better choice.
"""
import os
from functools import lru_cache

RERANKER_BACKEND = os.environ.get("RERANKER_BACKEND", "cross_encoder").lower()

# ms-marco-MiniLM-L-6-v2 is the standard lightweight cross-encoder for this task:
# fine-tuned on MS MARCO passage ranking (real IR dataset), ~66MB, CPU-runnable,
# output is a relevance logit (higher = more relevant, no fixed scale).
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _load_cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(DEFAULT_CROSS_ENCODER_MODEL)


def rerank(query: str, results: list, top_k: int | None = None) -> list:
    """Rerank a list of SearchResult objects using the cross-encoder.

    Returns the same list sorted by cross-encoder relevance score (descending),
    trimmed to top_k if given. Falls back to the original order if
    RERANKER_BACKEND=none or if the cross-encoder fails to load.

    The cross-encoder score replaces similarity_score in the returned results
    so downstream callers see a unified score regardless of backend.
    """
    if not results:
        return results

    if RERANKER_BACKEND == "none":
        return results[:top_k] if top_k else results

    try:
        model = _load_cross_encoder()
        pairs = [(query, r.text) for r in results]
        scores = model.predict(pairs)

        scored = sorted(
            zip(scores, results),
            key=lambda t: float(t[0]),
            reverse=True,
        )

        reranked = []
        for score, result in scored:
            # Replace similarity_score with cross-encoder score so the LLM context
            # builder and evaluation scripts see a single unified relevance signal.
            from dataclasses import replace
            reranked.append(replace(result, similarity_score=round(float(score), 4)))

        return reranked[:top_k] if top_k else reranked

    except Exception:
        # Cross-encoder unavailable (torch not installed, model download failed, etc.)
        # Fall back to original order rather than crashing the retrieval pipeline.
        return results[:top_k] if top_k else results
