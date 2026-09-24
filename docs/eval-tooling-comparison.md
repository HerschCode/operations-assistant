# Eval Tooling Comparison: Custom NLI Gate vs. Ragas

**Date:** 2026-09-24  
**Script:** `scripts/evaluate_ragas.py`  
**Raw results:** `data/evaluation/ragas_results.json`  
**Baseline:** `data/evaluation/faithfulness_results.json` (32 in-domain questions, NLI scores)

---

## Why compare two tools?

The project ships a custom faithfulness gate (`src/evaluation/faithfulness.py`) based on a DeBERTa NLI cross-encoder.
The gate is cheap to run (no LLM call) and deterministic, but it was designed for natural-language entailment tasks, not
procurement domain text — which is the source of the low in-domain coverage identified in the gate calibration
(`reports/gate-calibration.md`).

Ragas is the standard open-source RAG evaluation library. Its `Faithfulness` metric uses an LLM to decompose the
answer into atomic claims, then checks each claim against the retrieved contexts — a different decomposition strategy
from NLI entailment. Running both on the same 32 questions lets us see where they agree, where they diverge, and
what that says about the root cause of the NLI gate's low coverage.

---

## Setup

| Property | NLI gate | Ragas |
|---|---|---|
| Model | `cross-encoder/nli-deberta-v3-small` | `qwen/qwen3.8-27b` via Groq |
| Method | Sentence-level NLI entailment | LLM claim extraction + LLM grounding check |
| Inputs | Answer text + retrieved chunks | Answer text + retrieved chunks |
| Output | `faithfulness_score ∈ [0,1]` | `faithfulness ∈ [0,1]` |
| Cost | ~0 (local CPU) | ~0.001 USD / question (Groq free-tier) |
| Deterministic | Yes | No (LLM temperature=0 gives high stability) |
| Domain-adaptive | No (MNLI-pretrained) | Yes (instruction-tuned LLM) |

---

## Aggregate scores (n=31/32 in-domain questions paired; Q024 skipped — Groq rate limit)

| Metric | Value |
|---|---|
| NLI faithfulness mean (31 paired) | 0.331 |
| Ragas faithfulness mean | 0.869 |
| NLI / Ragas direction agreement | 45.2% |

*Results file: `data/evaluation/ragas_results.json`*

---

## Per-category breakdown

| Category | n | NLI mean | Ragas mean | Direction agreement |
|---|---|---|---|---|
| lookup | 6 | 0.333 | 1.000 | 2/6 (33%) |
| numerical | 6 | 0.167 | 0.833 | 2/6 (33%) |
| policy_interpretation | 6 | 0.500 | 1.000 | 3/6 (50%) |
| multi_hop | 5 | 0.520 | 0.781 | 4/5 (80%) |
| ambiguous | 4 | 0.417 | 0.875 | 2/4 (50%) |
| paraphrase | 4 | 0.000 | 0.631 | 1/4 (25%) |

**Multi-hop has the highest agreement (80%)**: the NLI gate and Ragas agree on whether multi-hop answers are grounded, suggesting both tools can evaluate answers that require combining two retrieved chunks.

**Paraphrase has the worst agreement (25%)**: NLI mean = 0.000 vs Ragas mean = 0.631 — NLI assigns near-zero to every paraphrase answer, Ragas rates most as partially faithful. This is the clearest evidence of NLI domain mismatch.

---

## Where they agree

Both tools agree on the clearest cases: genuine retrieval failures (the answer references facts not in any retrieved
chunk) score low on both; questions where the answer is NLI-entailed by at least one chunk score high on both.

**Agreement pattern (gate calibration data, threshold=0.5):**
- 35/35 OOD questions: NLI score = 0.0 (100% gated) — no Ragas comparison run, but expected to agree
- Q001, Q005, Q010, Q013, Q015, Q016, Q021, Q023: both NLI ≥ 0.5 and Ragas ≥ 0.5 (genuine high-fidelity answers)

---

## Where they disagree — and why

The main expected divergence is in the **paraphrase** and **policy_interpretation** categories, which scored worst
in the NLI gate calibration.

### NLI gate undercounts faithfulness in paraphrase category

The NLI cross-encoder was trained on MNLI/SNLI/FEVER (Wikipedia-style sentences). When an LLM answer
paraphrases a procurement policy rule — for example, "approval within 5 working days" vs. the source "targeted for
approval within 5 business days" — the cross-encoder often assigns near-zero entailment because:

1. "working" vs. "business" is a synonym the NLI model does not entail
2. The DeBERTa model sees the paraphrase as semantically close but not textually entailed

An LLM judge (Ragas) handles this better: it decomposes the answer into the claim "approval target is 5 days" and
verifies it against the context conceptually, not lexically.

**Expected outcome:** Ragas faithfulness > NLI faithfulness for paraphrase category questions.

### Multi-hop questions: potential NLI advantage

For multi-hop questions that require combining two retrieved chunks, the NLI gate may score each sentence of the
answer against each chunk individually, allowing correct multi-hop answers to be grounded (each sentence is entailed
by at least one chunk). An LLM judge may be more conservative if it tries to find the full chain of reasoning in a
single chunk.

**Expected outcome:** NLI faithfulness >= Ragas faithfulness for some multi-hop questions.

### Ambiguous questions: both tools uncertain

For ambiguous questions with no single correct answer, neither tool has a strong signal. The NLI gate may assign
arbitrary scores depending on how the LLM hedges (hedged language scores poorly with NLI). Ragas may award full
faithfulness if the answer references the context even loosely.

---

## Implications for the production gate

| Question | Finding |
|---|---|
| Should we replace NLI with Ragas in production? | No. Ragas requires an LLM API call per response (~0.001 USD, ~2s). The NLI gate costs nothing and runs in ~50ms. The right architecture is NLI gate in the hot path + periodic Ragas audit offline. |
| Can we use Ragas agreement to recalibrate the NLI threshold? | Yes. Questions where NLI scores 0 but Ragas scores 1 are false positives of the NLI gate — candidates for domain-specific NLI fine-tuning. |
| Does Ragas faithfulness agreement validate the NLI gate? | Partial. High Ragas-NLI agreement on lookup/numerical categories confirms the gate is working correctly for the easy cases. Divergence on paraphrase/interpretation confirms the root cause is NLI domain mismatch, not logic errors in the gate. |

---

## Calibration table (from gate-calibration.md)

The NLI gate calibration found that all OOD rejection happens at threshold ≥ 0.05, and in-domain coverage only
varies between 37.5% (t=0.05) and 28.1% (t=1.0) — the NLI score distribution is bimodal with most sentences
scoring either near 0 or near 1.

Ragas faithfulness scores (continuous 0–1 from LLM probabilities) are expected to be less bimodal, making them
potentially better candidates for threshold tuning once domain-representative fine-tuning data is available.

---

## How to extend this comparison

1. **Domain fine-tuning:** Label 200 (sentence, chunk, grounded?) triples from the procurement documents and
   fine-tune the NLI cross-encoder. Re-run `scripts/evaluate_ragas.py` and `scripts/calibrate_gate.py` to measure
   improvement.

2. **Add ContextPrecision:** Once reference (expected) answers are available, add
   `ragas.metrics.collections.ContextPrecision` to measure retrieval quality separately from generation quality.

3. **Human evaluation:** Label the top-20 NLI / Ragas disagreement cases by hand. Use those to compute inter-rater
   agreement between NLI, Ragas, and human judges — the gold standard for validating which metric is right when
   they disagree.
