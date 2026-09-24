"""
Tests for src/evaluation/faithfulness.py.

All NLI model calls are mocked — tests verify the scoring logic, sentence
splitting, aggregation, and fallback behavior, not the model weights.
"""
from unittest.mock import patch, MagicMock

import pytest

import src.evaluation.faithfulness as faith_module
from src.evaluation.faithfulness import (
    score_faithfulness,
    _split_sentences,
    FaithfulnessResult,
    ENTAILMENT_THRESHOLD,
)


# ── sentence splitting ────────────────────────────────────────────────────────

def test_split_sentences_basic():
    text = "The policy requires approval. Amounts above $10,000 need dual sign-off."
    sentences = _split_sentences(text)
    assert len(sentences) == 2


def test_split_sentences_filters_short_fragments():
    text = "Yes. The approval threshold is $5,000 for standard purchases."
    sentences = _split_sentences(text)
    # "Yes." is too short (< 20 chars) and should be filtered out
    assert all(len(s) > 20 for s in sentences)


def test_split_sentences_empty_string():
    assert _split_sentences("") == []


# ── backend=none ─────────────────────────────────────────────────────────────

def test_score_faithfulness_none_backend_returns_null_result():
    original = faith_module.FAITHFULNESS_BACKEND
    faith_module.FAITHFULNESS_BACKEND = "none"
    try:
        result = score_faithfulness("Some answer.", ["Some context."])
        assert result.backend_used == "none"
        assert result.faithfulness_score == 0.0
    finally:
        faith_module.FAITHFULNESS_BACKEND = original


def test_score_faithfulness_empty_chunks_returns_null_result():
    result = score_faithfulness("Some answer.", [])
    assert result.faithfulness_score == 0.0


# ── NLI scoring logic ─────────────────────────────────────────────────────────

def _mock_nli_scores(entailment_probs: list[float], n_chunks: int = 1):
    """Return mock NLI softmax outputs: [[contradiction, entailment, neutral]] per pair.
    Splits entailment probability evenly, sets contradiction to 0.1, neutral fills rest."""
    import numpy as np
    rows = []
    for ent_prob in entailment_probs:
        for _ in range(n_chunks):
            rows.append([0.1, ent_prob, max(0.0, 0.9 - ent_prob)])
    return np.array(rows)


@patch("src.evaluation.faithfulness._load_nli_model")
def test_score_faithfulness_all_entailed(mock_load):
    faith_module.FAITHFULNESS_BACKEND = "nli"
    faith_module._load_nli_model.cache_clear()

    mock_model = MagicMock()
    # 2 sentences × 1 chunk → 2 pairs, both highly entailed
    mock_model.predict.return_value = _mock_nli_scores([0.9, 0.85])
    mock_load.return_value = mock_model

    try:
        answer = "All purchase orders require approval. Amounts above $5,000 need dual sign-off."
        result = score_faithfulness(answer, ["Policy text about approval thresholds."])
        assert result.faithfulness_score == 1.0
        assert result.n_grounded == result.n_sentences
        assert result.backend_used == "nli"
    finally:
        faith_module._load_nli_model.cache_clear()


@patch("src.evaluation.faithfulness._load_nli_model")
def test_score_faithfulness_partial_entailment(mock_load):
    faith_module.FAITHFULNESS_BACKEND = "nli"
    faith_module._load_nli_model.cache_clear()

    mock_model = MagicMock()
    # 2 sentences: first entailed, second not
    mock_model.predict.return_value = _mock_nli_scores([0.85, 0.1])
    mock_load.return_value = mock_model

    try:
        answer = "The approval threshold is clearly stated. This was invented by the model."
        result = score_faithfulness(answer, ["Context about approval only."])
        assert result.n_sentences == 2
        assert result.n_grounded == 1
        assert result.faithfulness_score == pytest.approx(0.5, abs=0.01)
    finally:
        faith_module._load_nli_model.cache_clear()


@patch("src.evaluation.faithfulness._load_nli_model")
def test_score_faithfulness_detects_contradiction(mock_load):
    faith_module.FAITHFULNESS_BACKEND = "nli"
    faith_module._load_nli_model.cache_clear()

    import numpy as np
    mock_model = MagicMock()
    # High contradiction score for one sentence
    mock_model.predict.return_value = np.array([[0.7, 0.1, 0.2]])  # contradiction=0.7
    mock_load.return_value = mock_model

    try:
        answer = "The threshold is actually $1,000 for all orders."
        result = score_faithfulness(answer, ["The threshold is $10,000 for standard orders."])
        assert result.n_contradicted >= 1
        assert result.contradiction_rate > 0
    finally:
        faith_module._load_nli_model.cache_clear()


@patch("src.evaluation.faithfulness._load_nli_model")
def test_score_faithfulness_falls_back_on_exception(mock_load):
    faith_module.FAITHFULNESS_BACKEND = "nli"
    faith_module._load_nli_model.cache_clear()
    mock_load.side_effect = RuntimeError("torch not available")

    try:
        result = score_faithfulness(
            "The policy approval threshold for standard orders is clearly defined in section 4.",
            ["Any context chunk about policy."],
        )
        assert "error" in result.backend_used
        assert result.faithfulness_score == 0.0
    finally:
        faith_module._load_nli_model.cache_clear()


def test_faithfulness_result_is_faithful_threshold():
    r = FaithfulnessResult(faithfulness_score=0.85, contradiction_rate=0.0)
    assert r.is_faithful

    r2 = FaithfulnessResult(faithfulness_score=0.6, contradiction_rate=0.1)
    assert not r2.is_faithful


# ── empty-response regression (bug fix 2026-09-23) ───────────────────────────
# Before the fix, score_faithfulness("", chunks) returned faithfulness_score=1.0
# because 0 sentences were checked and the code assumed "nothing to check = fully
# faithful". This allowed empty LLM responses to bypass the gate. The correct
# treatment is score=0.0 so the gate fires.

def test_empty_answer_returns_zero_faithfulness():
    """Empty answer must score 0.0, not 1.0 (regression for empty-response bypass)."""
    result = score_faithfulness("", ["Any retrieved chunk about procurement policy."])
    assert result.faithfulness_score == 0.0, (
        f"Expected 0.0, got {result.faithfulness_score}. "
        "Empty responses must gate, not pass."
    )
    assert result.n_sentences == 0


def test_whitespace_only_answer_returns_zero_faithfulness():
    result = score_faithfulness("   \n  ", ["Some context chunk."])
    assert result.faithfulness_score == 0.0
    assert result.n_sentences == 0


def test_short_only_answer_returns_zero_faithfulness():
    """Answers that consist only of short fragments (< 20 chars) should also score 0."""
    result = score_faithfulness("Yes.", ["Context text about approval policy."])
    assert result.faithfulness_score == 0.0
    assert result.n_sentences == 0
