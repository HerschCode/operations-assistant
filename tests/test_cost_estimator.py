import pytest
from src.evaluation.cost_estimator import (
    estimate_tokens, calculate_cost, estimate_cost_from_text, load_pricing,
)


def test_load_pricing_reads_real_config():
    pricing = load_pricing()
    assert "claude-sonnet-4-6" in pricing["models"]
    assert pricing["default_model"] == "claude-sonnet-4-6"


def test_estimate_tokens_rough_heuristic():
    assert estimate_tokens("a" * 400) == 100  # 400 chars / 4
    assert estimate_tokens("") == 1  # never zero, avoids a zero-cost illusion on empty input


def test_calculate_cost_matches_hand_math_for_known_rates():
    # sonnet-4-6: $3/M input, $15/M output (config/pricing.yaml)
    # 1,000,000 input tokens -> exactly $3.00; 1,000,000 output tokens -> exactly $15.00
    result = calculate_cost(input_tokens=1_000_000, output_tokens=1_000_000, model="claude-sonnet-4-6")
    assert result.input_cost_usd == pytest.approx(3.00)
    assert result.output_cost_usd == pytest.approx(15.00)
    assert result.total_cost_usd == pytest.approx(18.00)


def test_calculate_cost_applies_batch_discount():
    full = calculate_cost(input_tokens=1_000_000, output_tokens=1_000_000, model="claude-sonnet-4-6", batch=False)
    batched = calculate_cost(input_tokens=1_000_000, output_tokens=1_000_000, model="claude-sonnet-4-6", batch=True)
    assert batched.total_cost_usd == pytest.approx(full.total_cost_usd * 0.5)


def test_calculate_cost_raises_on_unknown_model():
    with pytest.raises(ValueError, match="No pricing configured"):
        calculate_cost(input_tokens=100, output_tokens=100, model="not-a-real-model")


def test_estimate_cost_from_text_marks_estimated_flag():
    result = estimate_cost_from_text("some input text here", "some output text here")
    assert result.is_estimated_token_count is True
    assert result.total_cost_usd > 0


def test_calculate_cost_with_real_usage_marks_not_estimated():
    result = calculate_cost(input_tokens=500, output_tokens=200, is_estimated_token_count=False)
    assert result.is_estimated_token_count is False
