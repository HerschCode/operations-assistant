"""
Cost estimation, built from two genuinely different kinds of number:
1. Pricing rates (config/pricing.yaml) -- real published rates, with an explicit
   sourcing caveat, not fabricated.
2. Token counts -- when a real API response is available (it carries actual usage
   counts), those are exact. Without one, `estimate_tokens()` provides a rough
   chars/4 heuristic for pre-flight planning ONLY -- clearly labeled as an estimate,
   never presented as a measured figure. This project has no real API usage yet
   (every test mocks the LLM client), so nothing here claims to report real spend --
   it's a calculator, ready to use the moment real usage numbers exist.
"""
from dataclasses import dataclass
from pathlib import Path

import yaml

CHARS_PER_TOKEN_ESTIMATE = 4  # rough heuristic, not a real tokenizer -- see module docstring


def load_pricing(path: str = "config/pricing.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def estimate_tokens(text: str) -> int:
    """Rough pre-flight estimate only (chars / 4) -- NOT a real tokenizer count.
    Use actual `usage.input_tokens`/`usage.output_tokens` from a real API response
    when one is available; this exists for planning before any real call is made."""
    return max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)


@dataclass
class CostEstimate:
    model: str
    input_tokens: int
    output_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    is_estimated_token_count: bool  # True if token counts came from estimate_tokens(), not real usage


def calculate_cost(
    input_tokens: int, output_tokens: int, model: str | None = None,
    pricing: dict | None = None, batch: bool = False, is_estimated_token_count: bool = True,
) -> CostEstimate:
    pricing = pricing or load_pricing()
    model = model or pricing["default_model"]

    if model not in pricing["models"]:
        raise ValueError(f"No pricing configured for model '{model}' -- add it to config/pricing.yaml")

    rates = pricing["models"][model]
    discount = pricing.get("batch_discount", 1.0) if batch else 1.0

    input_cost = (input_tokens / 1_000_000) * rates["input_per_million"] * discount
    output_cost = (output_tokens / 1_000_000) * rates["output_per_million"] * discount

    return CostEstimate(
        model=model, input_tokens=input_tokens, output_tokens=output_tokens,
        input_cost_usd=round(input_cost, 6), output_cost_usd=round(output_cost, 6),
        total_cost_usd=round(input_cost + output_cost, 6),
        is_estimated_token_count=is_estimated_token_count,
    )


def estimate_cost_from_text(
    input_text: str, output_text: str, model: str | None = None,
    pricing: dict | None = None, batch: bool = False,
) -> CostEstimate:
    """Convenience wrapper for pre-flight planning -- estimates tokens from raw text
    via the chars/4 heuristic, then prices them. Always sets
    is_estimated_token_count=True since there's no real usage number involved."""
    return calculate_cost(
        input_tokens=estimate_tokens(input_text), output_tokens=estimate_tokens(output_text),
        model=model, pricing=pricing, batch=batch, is_estimated_token_count=True,
    )


if __name__ == "__main__":
    example = estimate_cost_from_text(
        input_text="What is the average procurement cycle time?" * 1,
        output_text="The average cycle time across all cases is 48.3 hours." * 1,
    )
    print(f"Rough pre-flight estimate for one simple chat turn ({example.model}):")
    print(f"  ~{example.input_tokens} input tokens, ~{example.output_tokens} output tokens")
    print(f"  ~${example.total_cost_usd:.6f} (estimated, not measured -- see module docstring)")
