"""
Top-K semantic search against the Chroma collection built in Phase 14. Deliberately
thin -- embed the query, query the collection, filter by min_similarity_score, shape
the result for citation use. The hard parts (chunking, embedding, storage) already
exist; this file is mostly wiring, as flagged in PLAN.md before it was written.
"""
from dataclasses import dataclass

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
