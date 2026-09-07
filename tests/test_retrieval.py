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
