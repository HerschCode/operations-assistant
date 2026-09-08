"""
Real BM25Okapi runs against small fixture corpora (rank_bm25 itself isn't mocked --
only Chroma's collection.query()/get() and the embedding call are, same principle
as test_retrieval.py's semantic_search tests). What's under test is the merge logic
(Reciprocal Rank Fusion combining semantic + keyword rankings), not whether BM25's
math is correct -- that's the library's own job.
"""
from unittest.mock import patch, MagicMock

from src.retrieval.search import hybrid_search


def _doc(document_id, title, section, chunk_index):
    return {"document_id": document_id, "title": title, "section_title": section, "chunk_index": chunk_index}


CORPUS = [
    ("PO Change Approval requires a manager signature within 2 business days.", _doc("proc-policy", "Procurement Policy", "6. PO Change Approval", 0)),
    ("The company values transparency and efficiency in all procurement activities.", _doc("proc-policy", "Procurement Policy", "1. Purpose", 1)),
    ("Cycle time is measured from Purchase Requisition creation to final Payment.", _doc("sla-policy", "SLA Policy", "3. What Counts", 0)),
    ("Suppliers are evaluated quarterly on delivery performance.", _doc("sla-policy", "SLA Policy", "7. Supplier Review", 1)),
]


def _mock_collection(semantic_order: list[int]):
    """semantic_order: indices into CORPUS, in the order semantic search should
    rank them (first = most similar)."""
    mock_collection = MagicMock()
    mock_collection.get.return_value = {
        "documents": [c[0] for c in CORPUS],
        "metadatas": [c[1] for c in CORPUS],
    }
    mock_collection.query.return_value = {
        "documents": [[CORPUS[i][0] for i in semantic_order]],
        "metadatas": [[CORPUS[i][1] for i in semantic_order]],
        # distances chosen so _distance_to_similarity gives a reasonable spread,
        # all above a typical min_similarity threshold
        "distances": [[0.2 * (rank + 1) for rank in range(len(semantic_order))]],
    }
    return mock_collection


@patch("src.retrieval.search.embed_query")
@patch("src.retrieval.search.get_collection")
def test_exact_keyword_match_surfaces_even_if_semantic_ranks_it_low(mock_get_collection, mock_embed):
    """The actual point of hybrid retrieval: a chunk containing the query's exact
    term ("PO Change Approval") should rank near the top via BM25, even when
    semantic search (mocked here to rank it LAST) would have buried it."""
    mock_embed.return_value = [0.1, 0.2, 0.3]
    # semantic ranks index 0 (the PO Change Approval chunk) dead last
    mock_get_collection.return_value = _mock_collection(semantic_order=[1, 3, 2, 0])

    results = hybrid_search("PO Change Approval", top_k=4, min_similarity=0.0)

    result_texts = [r.text for r in results]
    assert CORPUS[0][0] in result_texts[:2]  # surfaces near the top via keyword match


@patch("src.retrieval.search.embed_query")
@patch("src.retrieval.search.get_collection")
def test_semantic_only_match_still_surfaces(mock_get_collection, mock_embed):
    """A chunk that's the top semantic result but shares no distinctive keywords
    with the query should still surface -- hybrid retrieval adds keyword signal,
    it doesn't replace semantic signal."""
    mock_embed.return_value = [0.1, 0.2, 0.3]
    mock_get_collection.return_value = _mock_collection(semantic_order=[2, 0, 1, 3])

    results = hybrid_search("when does the clock start running", top_k=2, min_similarity=0.0)

    result_texts = [r.text for r in results]
    assert CORPUS[2][0] in result_texts  # the SLA "what counts" chunk, top semantic match


@patch("src.retrieval.search.embed_query")
@patch("src.retrieval.search.get_collection")
def test_respects_top_k(mock_get_collection, mock_embed):
    mock_embed.return_value = [0.1, 0.2, 0.3]
    mock_get_collection.return_value = _mock_collection(semantic_order=[0, 1, 2, 3])

    results = hybrid_search("procurement", top_k=2, min_similarity=0.0)
    assert len(results) == 2


@patch("src.retrieval.search.embed_query")
@patch("src.retrieval.search.get_collection")
def test_raises_on_empty_query(mock_get_collection, mock_embed):
    import pytest
    with pytest.raises(ValueError):
        hybrid_search("")
