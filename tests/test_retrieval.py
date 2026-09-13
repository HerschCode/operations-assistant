"""
Mocks embed_query and get_collection rather than hitting the real sentence-transformers
model / Chroma DB -- this keeps the suite fast and runnable without a model download or
persisted vector data, same principle as operations-performance's test_api.py mocking
DB calls. The logic under test here (similarity conversion, min_similarity filtering,
citation formatting) doesn't depend on the real embedding values, just on the shape of
what Chroma returns.
"""
from unittest.mock import patch, MagicMock
import pytest

from src.retrieval.search import (
    semantic_search,
    _distance_to_similarity,
    SearchResult,
    no_relevant_results_response,
    reranked_search,
)


def test_distance_to_similarity_identical_vectors():
    assert _distance_to_similarity(0.0) == 1.0


def test_distance_to_similarity_opposite_vectors():
    assert _distance_to_similarity(2.0) == 0.0


def test_distance_to_similarity_never_negative():
    assert _distance_to_similarity(3.0) == 0.0  # clamped, not -0.5


def test_search_result_citation_with_section():
    result = SearchResult(
        document_id="procurement-policy", title="Procurement Policy",
        section_title="4.2 Secondary Approval", text="...", similarity_score=0.8,
    )
    assert result.citation == "Procurement Policy, Section 4.2 Secondary Approval"


def test_search_result_citation_without_section():
    result = SearchResult(
        document_id="procurement-policy", title="Procurement Policy",
        section_title=None, text="...", similarity_score=0.8,
    )
    assert result.citation == "Procurement Policy"


def test_semantic_search_raises_on_empty_query():
    with pytest.raises(ValueError):
        semantic_search("")
    with pytest.raises(ValueError):
        semantic_search("   ")


@patch("src.retrieval.search.embed_query")
@patch("src.retrieval.search.get_collection")
def test_semantic_search_filters_below_min_similarity(mock_get_collection, mock_embed):
    mock_embed.return_value = [0.1, 0.2, 0.3]
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["highly relevant text", "barely relevant text"]],
        "metadatas": [[
            {"document_id": "doc1", "title": "Doc One", "section_title": "1. Intro"},
            {"document_id": "doc2", "title": "Doc Two", "section_title": "2. Other"},
        ]],
        "distances": [[0.1, 1.9]],  # first is very similar, second is nearly opposite
    }
    mock_get_collection.return_value = mock_collection

    results = semantic_search("some query", min_similarity=0.3)

    assert len(results) == 1  # the low-similarity result gets filtered out
    assert results[0].document_id == "doc1"


@patch("src.retrieval.search.embed_query")
@patch("src.retrieval.search.get_collection")
def test_semantic_search_returns_empty_list_when_nothing_above_threshold(mock_get_collection, mock_embed):
    mock_embed.return_value = [0.1, 0.2, 0.3]
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["irrelevant text"]],
        "metadatas": [[{"document_id": "doc1", "title": "Doc One", "section_title": None}]],
        "distances": [[1.9]],
    }
    mock_get_collection.return_value = mock_collection

    results = semantic_search("unrelated query", min_similarity=0.3)
    assert results == []
    assert "No document" in no_relevant_results_response()


@patch("src.retrieval.search.embed_query")
@patch("src.retrieval.search.get_collection")
def test_semantic_search_respects_top_k(mock_get_collection, mock_embed):
    mock_embed.return_value = [0.1, 0.2, 0.3]
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "documents": [["a", "b", "c"]],
        "metadatas": [[
            {"document_id": f"doc{i}", "title": f"Doc {i}", "section_title": None}
            for i in range(3)
        ]],
        "distances": [[0.1, 0.2, 0.3]],
    }
    mock_get_collection.return_value = mock_collection

    semantic_search("query", top_k=3)
    mock_collection.query.assert_called_once()
    assert mock_collection.query.call_args.kwargs["n_results"] == 3


# ── reranker tests ───────────────────────────────────────────────────────────

def _make_results(n: int) -> list[SearchResult]:
    return [
        SearchResult(
            document_id=f"doc{i}", title=f"Doc {i}", section_title=None,
            text=f"text chunk number {i}", similarity_score=float(n - i) / n,
        )
        for i in range(n)
    ]


def test_reranker_none_backend_returns_original_order():
    """RERANKER_BACKEND=none must preserve the original hybrid order."""
    import src.retrieval.reranker as rr
    original = rr.RERANKER_BACKEND
    rr.RERANKER_BACKEND = "none"
    try:
        results = _make_results(5)
        reranked = rr.rerank("any query", results, top_k=3)
        assert [r.document_id for r in reranked] == ["doc0", "doc1", "doc2"]
    finally:
        rr.RERANKER_BACKEND = original


def test_reranker_cross_encoder_reorders_by_score():
    """With a mocked cross-encoder, results should be re-sorted by CE score."""
    import src.retrieval.reranker as rr
    original = rr.RERANKER_BACKEND
    rr.RERANKER_BACKEND = "cross_encoder"
    try:
        results = _make_results(3)  # doc0, doc1, doc2 in that order
        # mock: CE thinks doc2 is most relevant, then doc0, then doc1
        with patch("src.retrieval.reranker._load_cross_encoder") as mock_load:
            mock_model = MagicMock()
            mock_model.predict.return_value = [0.5, 0.1, 0.9]  # scores for doc0, doc1, doc2
            mock_load.return_value = mock_model
            rr._load_cross_encoder.cache_clear()
            reranked = rr.rerank("query", results, top_k=3)
        assert reranked[0].document_id == "doc2"  # highest CE score
        assert reranked[1].document_id == "doc0"
        assert reranked[2].document_id == "doc1"
    finally:
        rr.RERANKER_BACKEND = original
        rr._load_cross_encoder.cache_clear()


def test_reranker_falls_back_on_exception():
    """If the CE model raises, rerank() returns the original order rather than crashing."""
    import src.retrieval.reranker as rr
    original = rr.RERANKER_BACKEND
    rr.RERANKER_BACKEND = "cross_encoder"
    try:
        results = _make_results(3)
        with patch("src.retrieval.reranker._load_cross_encoder", side_effect=RuntimeError("no torch")):
            rr._load_cross_encoder.cache_clear()
            reranked = rr.rerank("query", results)
        assert [r.document_id for r in reranked] == ["doc0", "doc1", "doc2"]
    finally:
        rr.RERANKER_BACKEND = original
        rr._load_cross_encoder.cache_clear()


def test_reranked_search_calls_hybrid_then_rerank():
    """reranked_search() should call hybrid_search with candidate_k then rerank the candidates."""
    with patch("src.retrieval.search.hybrid_search") as mock_hybrid, \
         patch("src.retrieval.reranker.rerank") as mock_rerank:
        mock_hybrid.return_value = _make_results(5)
        mock_rerank.return_value = _make_results(3)
        result = reranked_search("test query", top_k=3, candidate_k=10)
        mock_hybrid.assert_called_once()
        # candidate_k goes to hybrid_search's top_k param
        call_kwargs = mock_hybrid.call_args.kwargs
        call_args = mock_hybrid.call_args.args
        assert call_kwargs.get("top_k", call_args[1] if len(call_args) > 1 else None) == 10
        assert len(result) == 3
