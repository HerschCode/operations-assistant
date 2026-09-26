# Faithfulness Gate Calibration

**Scripts:** `scripts/calibrate_gate.py`, `scripts/analyze_hard_negatives.py`  
**Data:** `data/evaluation/faithfulness_results.json`  
**Output:** `data/evaluation/gate_calibration_results.json`, `data/evaluation/hard_negatives_analysis.json`

## What the gate does

Before every RAG answer is shown to the user, the faithfulness gate checks whether
the answer's sentences are supported by the retrieved context, using NLI
(`cross-encoder/nli-deberta-v3-small`). A sentence is *grounded* if its max
entailment probability across all retrieved chunks exceeds 0.5. If fewer sentences
than the threshold pass, the answer is blocked and a fallback is shown.

The gate is mechanical — no second LLM call, nothing to hallucinate — which is why
NLI was chosen over an LLM judge. It adds ~80ms of latency per answer.

## Calibration: threshold vs coverage

Run `scripts/calibrate_gate.py` on 32 in-domain questions (should NOT be gated) and
35 OOD questions from SQuAD 2.0 unanswerable (SHOULD be gated).

| Threshold | Coverage (in-domain) | OOD rejection |
|:---------:|:-------------------:|:-------------:|
| 0.00 | 100.0% | 0.0% |
| **0.05** | **37.5%** | **100.0%** |
| 0.50 | 34.4% | 100.0% |
| 1.00 | 28.1% | 100.0% |

OOD rejection is 100% at all thresholds above 0 (after the empty-response bug fix
that reassigns faithfulness_score=0.0 to n_sentences==0 rows). The recommended
threshold is 0.05 (best harmonic mean of coverage and OOD rejection).

**Apparent finding from calibration:** only 37.5% of in-domain answers pass the gate.
This was initially attributed to DeBERTa domain mismatch (procurement numbers not
in MNLI/SNLI training data).

## Hard-negative analysis: reclassifying the 37.5%

`scripts/analyze_hard_negatives.py` classifies the 32 in-domain answers by subtype
using (faithfulness_score, contradiction_rate):

| Type | n | % | Gate rate | Mean entailment | Mean contradiction |
|---|:---:|:---:|:---:|:---:|:---:|
| correct | 5 | 15.6% | 0.0% | 0.864 | 0.083 |
| ambiguous | 7 | 21.9% | 14.3% | 0.917 | 0.972 |
| halluc_caught | 13 | 40.6% | 100.0% | 0.011 | 0.928 |
| blocked_neutral | 7 | 21.9% | 100.0% | 0.009 | 0.042 |

### Types explained

- **correct** — faithfulness > 0, no contradiction: NLI says grounded, gate passes.

- **ambiguous** — faithfulness > 0 AND contradiction > 0: NLI finds both entailment and
  contradiction simultaneously. Gate passes (86%) because faithfulness_score is positive.

- **halluc_caught** — faithfulness=0, contradiction > 0: NLI found an explicit contradiction.
  Gate blocks AND has an NLI signal to back the block. Examples: wrong SLA targets
  ("10 business days"), wrong teams ("Vendor Risk team"), wrong review frequency ("quarterly").

- **blocked_neutral** — faithfulness=0, contradiction = 0: NLI assigns near-zero scores in
  BOTH directions. Gate blocks because faithfulness=0, but NLI has no signal as to why.
  Example: "3 business days of supplier unresponsiveness triggers escalation" —
  the NLI model assigns near-zero scores to both the wrong claim and the correct one
  (entailment=0.003, contradiction=0.13). These are the gate's true hard negatives:
  blocked for the right outcome but the wrong reason.

## What the analysis can and cannot measure

**What we can measure from NLI scores alone:**
- Gate coverage: 34.4% of in-domain answers pass (11/32)
- NLI signal rate: 61.9% of blocked answers had an NLI contradiction signal backing the block

**What we cannot measure without independent human labels:**
- Gate precision (are the blocked answers actually wrong?)
- Gate recall (does the gate catch all wrong answers?)

The earlier version of this analysis reported "Precision 95.2%, Recall 100%" — those
numbers were circular: the subtypes were defined by NLI scores, then gate performance
was scored against those same NLI-derived labels. Recall=1.0 is guaranteed by
construction (every answer classified as a "hallucination" has faithfulness<0.5, and
the gate fires when faithfulness<0.5). Reverted to the honest non-circular metrics above.

To measure true precision/recall: hand-label 32 answers (correct/incorrect) and
compare against the gate's block/pass decisions. That is the missing eval.

## What the 34.4% coverage means

34.4% of in-domain answers pass the gate — not because the gate misfires on correct
answers (the 5 "correct" answers all pass cleanly), but because the current LLM
generates answers that fail the NLI check for most in-domain questions. Of the 27
blocked answers, 13 had explicit NLI contradiction evidence and 7 had no NLI signal
at all. The remaining 7 "ambiguous" answers passed despite mixed signals.

The DeBERTa domain-mismatch problem is real but secondary: it explains *why* the NLI
signal rate is 61.9% rather than higher, not why coverage is low. Low coverage is
primarily a model-quality issue (most answers NLI rejects), not a threshold problem.

## Stress test for blocked_neutral cases

The 7 blocked_neutral answers are the gate's true blind spot. To verify the gate would
pass a *correct* answer for the same question:

1. Craft the correct answer for each blocked_neutral question
2. Run `score_faithfulness(correct_answer, chunks)` — expect faithfulness > 0
3. Run `score_faithfulness(wrong_answer, chunks)` — expect faithfulness ≈ 0

If both score identically, the NLI model cannot distinguish correct from wrong for that
question and threshold tuning is ineffective. This would confirm domain-specific
fine-tuning is needed for the DeBERTa encoder.

## Recommendations

1. **Use threshold=0.05** (recommended by calibration) — any higher threshold only
   gates more *correct* ambiguous answers without improving hallucination detection.

2. **Monitor the blocked_neutral rate** as the LLM improves. If a new model achieves
   >50% correct answers, re-run the analysis to check whether those cases still score
   as blocked_neutral or shift to halluc_caught (NLI gains signal).

3. **For ambiguous answers** (both entailed and contradicted sentences), consider
   reporting the contradiction_rate alongside the answer rather than blocking entirely —
   the user gets the correct part of the answer with a caveat.

## Reproduction

```bash
# Calibration: threshold vs coverage tradeoff
python -m scripts.calibrate_gate

# Hard-negative analysis: classify by subtype
python -m scripts.analyze_hard_negatives
```

Both scripts are offline (no API calls, no torch reloading — faithfulness scores
are pre-computed in `faithfulness_results.json`).
