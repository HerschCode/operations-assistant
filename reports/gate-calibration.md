# Faithfulness Gate Calibration

**Date:** 2026-09-23/24  
**Script:** `scripts/calibrate_gate.py`  
**Raw data:** `data/evaluation/gate_calibration_results.json`

## Labeled set

| Split | n | Ground truth |
|---|---|---|
| In-domain (faithfulness_results.json) | 32 | Should NOT be gated — answers exist in procurement docs |
| OOD (ood_abstention_results.json, post-fix) | 35 | Should be gated — SQuAD 2.0 unanswerable, no procurement overlap |
| **Total** | **67** | |

OOD scores corrected post-hoc for the empty-response bug: 28 rows where `n_sentences=0` are reassigned `faithfulness_score=0.0` (the corrected behaviour after the `faithfulness.py` fix; these were surfaced in the original eval with `faith=1.0` due to the bypass).

## Coverage vs. OOD rejection

| Threshold | In-domain coverage | OOD rejection |
|---|---|---|
| 0.00 | 100.0% | 0.0% |
| **0.05** | **37.5%** | **100.0%** |
| 0.10 | 37.5% | 100.0% |
| 0.20 | 34.4% | 100.0% |
| 0.50 (default) | 34.4% | 100.0% |
| 0.70 | 28.1% | 100.0% |
| 1.00 | 28.1% | 100.0% |

After the empty-response fix, OOD rejection is 100% at any threshold ≥ 0.05. Coverage ranges only from 37.5% (t=0.05) to 28.1% (t=1.0) — a 9-point window across the entire threshold range.

## Cost-ratio operating point

**Stated cost ratio:** surfacing an unfaithful answer costs 3× more than withholding a faithful one (`C_FN/C_FP = 3.0`). Rationale: a hallucinated answer that a user acts on causes a real operational error; an "insufficient information" response is inconvenient but harmless.

**Cost function:** `cost = FP_rate + 3 * FN_rate`  
where FP = in-domain question incorrectly gated, FN = OOD question incorrectly passed.

| Threshold | cost |
|---|---|
| 0.00 | 3.000 |
| **0.05** | **0.625** ← optimal |
| 0.50 (default) | 0.656 |
| 1.00 | 0.719 |

**Recommended operating point: t = 0.05.** Reduces cost by 4.8% vs the default t=0.5 (0.625 vs 0.656) by passing 3 more in-domain questions (37.5% vs 34.4% coverage) at identical OOD rejection.

## Claim-level vs. whole-answer scoring

The gate already uses claim-level (sentence-level) NLI: each sentence is scored against all retrieved chunks independently, and `faithfulness_score = n_grounded_sentences / n_sentences`. Whole-answer scoring (one NLI call for the full response) was not separately benchmarked because:

1. The DeBERTa cross-encoder has a 512-token input limit — long answers would be truncated, making it *worse* than sentence-level for multi-sentence responses.
2. The root cause of low coverage is not the aggregation strategy. The sentence-level max-entailment distribution is bimodal: 67% of in-domain answer sentences score `max_entailment < 0.05` and 33% score ≥ 0.5 (almost nothing in between). Switching to mean-pooling or whole-answer scoring cannot recover sentences that the NLI model rates near-zero.

**Root cause: NLI domain mismatch.** The `cross-encoder/nli-deberta-v3-small` model was trained on MNLI/SNLI/FEVER (Wikipedia-style English). Procurement document text uses domain jargon, table-formatted data, and LLM-paraphrased answers that the NLI model does not entail against verbatim source text.

**What would fix coverage:** (a) fine-tune the NLI model on (answer, chunk) pairs from this domain, or (b) replace NLI entailment with a chunk-citation check (require the LLM to cite the specific chunk and verify verbatim overlap), or (c) accept the low coverage as a conservative safety property of the current system.

## Summary

| Metric | Before fix | After fix |
|---|---|---|
| OOD bypass (empty response) | 28/35 (80%) | 0/35 (0%) |
| OOD rejection rate at t=0.5 | 20% | 100% |
| In-domain coverage at t=0.5 | 34.4% | 34.4% (unchanged) |
| Recommended threshold (cost ratio 3:1) | — | 0.05 |
| In-domain coverage at recommended t | — | 37.5% |
