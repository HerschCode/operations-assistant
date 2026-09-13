"""
Top-K semantic search against the Chroma collection built in Phase 14. Deliberately
thin -- embed the query, query the collection, filter by min_similarity_score, shape
the result for citation use. The hard parts (chunking, embedding, storage) already
exist; this file is mostly wiring, as flagged in PLAN.md before it was written.

hybrid_search() (below) adds keyword (BM25) retrieval alongside semantic search,
combined via Reciprocal Rank Fusion -- closes the FUTURE_IMPROVEMENTS.md gap
("Hybrid retrieval (keyword + semantic) and reranking, Tier 3 in FEATURES.md").
Semantic search alone misses exact-term matches a keyword search catches
naturally (e.g. a query using the literal term "PO Change Approval" should rank a
chunk containing that exact phrase highly even if its embedding similarity isn't
the top score) -- this project's own policy documents use precise defined terms
throughout, which is exactly the case hybrid retrieval helps with.
"""
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.retrieval.embeddings import embed_query
from src.retrieval.vector_store import get_collection

DEFAULT_TOP_K = 5
DEFAULT_MIN_SIMILARITY = 0.3  # from config/retrieval.yaml -- kept in sync manually for now
COLLECTION_NAME = "operations_policy_documents"


@dataclass
class SearchResult:
    document_id: str
    title: str
    section_title: str | None
    text: str
    similarity_score: float

    @property
    def citation(self) -> str:
        """The human-readable citation string used in agent answers, e.g.
        'Procurement Policy, Section 4.2' -- falls back to just the title if the
        chunk has no section (e.g. an unheadered intro paragraph)."""
        if self.section_title:
            return f"{self.title}, Section {self.section_title}"
        return self.title


def _distance_to_similarity(distance: float) -> float:
    """Chroma returns cosine DISTANCE by default (0 = identical, 2 = opposite);
    convert to a similarity score in [0, 1] so min_similarity_score in config reads
    naturally ('higher is more similar') rather than requiring the reader to know
    Chroma's distance convention."""
    return max(0.0, 1.0 - (distance / 2.0))


def semantic_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    collection_name: str = COLLECTION_NAME,
) -> list[SearchResult]:
    if not query or not query.strip():
        raise ValueError("query must not be empty")

    collection = get_collection(collection_name)
    query_embedding = embed_query(query)

    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    results = []
    documents = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    for text, metadata, distance in zip(documents, metadatas, distances):
        score = _distance_to_similarity(distance)
        if score < min_similarity:
            continue
        results.append(SearchResult(
            document_id=metadata["document_id"],
            title=metadata["title"],
            section_title=metadata.get("section_title") or None,
            text=text,
            similarity_score=round(score, 3),
        ))

    return results


RRF_K = 60  # standard Reciprocal Rank Fusion constant -- dampens the impact of rank
# 1 vs rank 2 being a huge score jump while rank 50 vs 51 is negligible; 60 is the
# widely-used default from the original RRF paper, not a value tuned for this corpus.


def _fetch_all_chunks(collection_name: str) -> list[dict]:
    """Pulls the full corpus (documents + metadata) for BM25 scoring. This
    project's document volume (a handful of policy documents, dozens of chunks --
    see docs/rag-design.md) makes scoring the whole corpus per query cheap; at a
    much larger corpus size, BM25 would need its own index rather than
    re-tokenizing every chunk on every call."""
    collection = get_collection(collection_name)
    raw = collection.get(include=["documents", "metadatas"])
    return [
        {"text": doc, "metadata": meta}
        for doc, meta in zip(raw.get("documents", []), raw.get("metadatas", []))
    ]


def hybrid_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    collection_name: str = COLLECTION_NAME,
    semantic_candidates: int = 20,
) -> list[SearchResult]:
    """Combines semantic search (existing embedding-based retrieval) with BM25
    keyword search over the same corpus, merged via Reciprocal Rank Fusion --
    a chunk's combined score is the sum of 1/(RRF_K + rank) across whichever
    method(s) it appeared in, so a chunk ranking well in EITHER method (or both)
    surfaces near the top, rather than requiring it to win on cosine similarity
    alone. min_similarity is still applied to the semantic ranking's own score
    (BM25 scores aren't on a comparable [0,1] scale, so filtering by that would be
    meaningless) -- a chunk can still surface via a strong keyword match even if
    its semantic similarity alone would have been filtered out."""
    if not query or not query.strip():
        raise ValueError("query must not be empty")

    collection = get_collection(collection_name)
    query_embedding = embed_query(query)
    semantic_raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=semantic_candidates,
        include=["documents", "metadatas", "distances"],
    )
    semantic_docs = semantic_raw.get("documents", [[]])[0]
    semantic_metas = semantic_raw.get("metadatas", [[]])[0]
    semantic_distances = semantic_raw.get("distances", [[]])[0]

    # Chunk identity for merging across the two ranked lists: (document_id, chunk_index)
    # is the same identifier index_documents.py already uses for Chroma's own chunk
    # IDs, stable and unique per chunk.
    def chunk_key(metadata: dict) -> tuple:
        return (metadata["document_id"], metadata.get("chunk_index"))

    semantic_rank: dict[tuple, int] = {}
    chunk_data: dict[tuple, dict] = {}
    for rank, (text, metadata, distance) in enumerate(zip(semantic_docs, semantic_metas, semantic_distances)):
        key = chunk_key(metadata)
        semantic_rank[key] = rank
        chunk_data[key] = {"text": text, "metadata": metadata, "similarity_score": _distance_to_similarity(distance)}

    all_chunks = _fetch_all_chunks(collection_name)
    if all_chunks:
        tokenized_corpus = [c["text"].lower().split() for c in all_chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(query.lower().split())
        bm25_ranked = sorted(range(len(all_chunks)), key=lambda i: bm25_scores[i], reverse=True)

        keyword_rank: dict[tuple, int] = {}
        for rank, idx in enumerate(bm25_ranked[:semantic_candidates]):
            chunk = all_chunks[idx]
            key = chunk_key(chunk["metadata"])
            keyword_rank[key] = rank
            if key not in chunk_data:
                chunk_data[key] = {"text": chunk["text"], "metadata": chunk["metadata"], "similarity_score": None}
    else:
        keyword_rank = {}

    rrf_scores: dict[tuple, float] = {}
    for key in set(semantic_rank) | set(keyword_rank):
        score = 0.0
        if key in semantic_rank:
            score += 1.0 / (RRF_K + semantic_rank[key])
        if key in keyword_rank:
            score += 1.0 / (RRF_K + keyword_rank[key])
        rrf_scores[key] = score

    ranked_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)

    results = []
    for key in ranked_keys[:top_k]:
        data = chunk_data[key]
        # A chunk found ONLY via keyword match (no semantic score) is still
        # included -- that's the actual point of hybrid retrieval -- but
        # min_similarity can only filter chunks that HAVE a semantic score.
        if data["similarity_score"] is not None and data["similarity_score"] < min_similarity:
            continue
        metadata = data["metadata"]
        results.append(SearchResult(
            document_id=metadata["document_id"],
            title=metadata["title"],
            section_title=metadata.get("section_title") or None,
            text=data["text"],
            similarity_score=round(data["similarity_score"], 3) if data["similarity_score"] is not None else 0.0,
        ))

    return results


def bm25_search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    collection_name: str = COLLECTION_NAME,
) -> list[SearchResult]:
    """BM25-only retrieval over the full chunk corpus. Used by the benchmark script
    to isolate keyword-search quality from semantic quality, so the two can be
    compared independently rather than only ever seeing the hybrid result."""
    if not query or not query.strip():
        raise ValueError("query must not be empty")

    all_chunks = _fetch_all_chunks(collection_name)
    if not all_chunks:
        return []

    tokenized_corpus = [c["text"].lower().split() for c in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(query.lower().split())
    ranked_idx = sorted(range(len(all_chunks)), key=lambda i: scores[i], reverse=True)

    results = []
    for idx in ranked_idx[:top_k]:
        chunk = all_chunks[idx]
        meta = chunk["metadata"]
        results.append(SearchResult(
            document_id=meta["document_id"],
            title=meta["title"],
            section_title=meta.get("section_title") or None,
            text=chunk["text"],
            similarity_score=round(float(scores[idx]), 4),
        ))
    return results


def no_relevant_results_response() -> str:
    """What the agent should say when semantic_search returns nothing above
    min_similarity -- a fixed, honest string rather than letting the LLM improvise
    an answer with no grounding. See docs/rag-design.md and the grounding rules
    that land in Phase 19."""
    return "No document in the policy corpus is relevant enough to answer this confidently."


if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "What approval is needed for high-value orders?"
    results = semantic_search(query)
    if not results:
        print(no_relevant_results_response())
    for r in results:
        print(f"[{r.similarity_score}] {r.citation}")
        print(f"  {r.text[:150]}...")
