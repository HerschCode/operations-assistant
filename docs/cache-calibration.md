# Semantic Cache Calibration

## Summary

The semantic cache (`src/cache/semantic_cache.py`) uses cosine-similarity gating to decide
whether a new query is close enough to a cached query to serve the stored response without
calling the LLM.

**At the deployed threshold of 0.97 (with `all-MiniLM-L6-v2`):**
- False-hit rate: **0.0%** — no false hits in the 15-pair test set
- Paraphrase recall: **0.0%** — no paraphrases hit either

The threshold is effectively a near-exact-match gate for this model. In practice the cache
only fires when a user repeats a question verbatim (or with trivial punctuation variation).

---

## Measurement

Evaluated with [`scripts/measure_cache_false_hit_rate.py`](../scripts/measure_cache_false_hit_rate.py)
on 15 hand-written question pairs (3 paraphrases, 12 different-intent):

```
Pair                                                         Similarity  Hit?
──────────────────────────────────────────────────────────────────────────────
Paraphrases (should hit)
  "What is the current SLA target?" / "What are our SLA targets?"   0.812  miss
  "How many POs are late?" / "What is the count of late POs?"        0.932  miss
  "Top 3 bottleneck suppliers?" / "Which 3 suppliers delay most?"    0.685  miss

Different-intent (should NOT hit — highest risks)
  Supplier A on-time rate / Supplier B on-time rate                  0.876  miss
  PO breach order 4 / PO breach order 12                             0.884  miss
  How many POs late this week / …last month                          0.831  miss
  Overall conformance / IT-segment conformance                       0.787  miss
```

Full results: [`data/evaluation/false_hit_rate_analysis.json`](../data/evaluation/false_hit_rate_analysis.json)

---

## Why 0.97 and what it means

The closest *paraphrase* pair reaches **0.932**. The closest *different-intent* pair reaches
**0.884**. There is a separation of **0.048** between the best-case paraphrase and the
worst-case false-hit candidate.

| Threshold | FHR | Paraphrase recall | Notes |
|---|---|---|---|
| **0.97** (deployed) | 0% | 0% | Only exact repetition hits |
| 0.94 | 0% | 0% | Still below all paraphrase sims |
| 0.93 | 0% | 33% | "count of late POs" hits; others miss |
| 0.90 | 0% | 33% | Safe on this set; 0.876 is closest false-hit candidate |
| 0.88 | ~8% | 33% | "Supplier A/B" pair would likely false-hit |

**What this means in production:** at 0.97, the cache is a safety-first warm-path for
*identical repeated questions* only. Users who type paraphrases always go to the LLM.
This is the correct default for a factual ops Q&A system where serving a stale answer
from a paraphrase of a different question would be worse than a slightly slower response.

**Operators who want paraphrase compression** should lower the threshold to ~0.93, which
catches the easiest paraphrases with no false-hit risk on this test set. Set:

```
SEMANTIC_CACHE_THRESHOLD=0.93
```

The risk at 0.93 is that question pairs with distinct slot values but identical framing
(e.g. "cycle time for IT" vs "cycle time for Facilities", sim=0.656) are still safe, but
novel questions outside the test set are not characterised. Measure on your own traffic.

---

## Caveats

- Test set is 15 hand-written pairs; real traffic will include question types not covered here.
- Similarity depends on the embedding model. If `SEMANTIC_CACHE_MODEL` is changed, re-run
  the calibration script before adjusting the threshold.
- The cache is disabled by default (`SEMANTIC_CACHE=1` to enable). The false-hit rate
  measurement was performed offline with no live LLM calls.
