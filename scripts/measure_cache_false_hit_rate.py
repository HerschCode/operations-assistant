"""
Measure the semantic cache false-hit rate at the deployed threshold (0.97).

A *false hit* is when two questions have different expected answers but the
cache considers them similar enough to serve the same cached response.

Method
------
1. Build a list of question pairs (Q_anchor, Q_probe) where a human expects
   different answers (different tools, different data, different scope).
2. For each pair: embed both; compute cosine similarity.
3. A false hit would occur if sim >= threshold *and* the answers genuinely differ.
4. Because we have no live answers to compare, we classify intent manually
   (tool_expected, expected_answer_differs = True) and report the cosine
   similarity distribution.

Output
------
- false_hit_rate_analysis.json  — machine-readable results
- Prints a summary table to stdout

Usage
-----
    python scripts/measure_cache_false_hit_rate.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

THRESHOLD = float(os.environ.get("SEMANTIC_CACHE_THRESHOLD", "0.97"))
MODEL_NAME = os.environ.get("SEMANTIC_CACHE_MODEL", "all-MiniLM-L6-v2")
OUT_PATH = Path("data/evaluation/false_hit_rate_analysis.json")

# ---------------------------------------------------------------------------
# Question pairs: (anchor, probe, expected_same_answer)
#
# expected_same_answer=True  → a hit here is correct (true positive)
# expected_same_answer=False → a hit here is a false positive (false hit)
# ---------------------------------------------------------------------------
PAIRS: list[tuple[str, str, bool, str]] = [
    # (anchor, probe, expected_same_answer, rationale)

    # --- True positives: paraphrases should hit --------------------------------
    ("What is the current SLA target?",
     "What are our SLA targets?",
     True, "paraphrase — same metric"),
    ("How many purchase orders are late?",
     "What is the count of late purchase orders?",
     True, "paraphrase — same metric"),
    ("Who are the top 3 bottleneck suppliers?",
     "Which 3 suppliers cause the most delays?",
     True, "paraphrase — same metric"),

    # --- False-hit candidates: look similar but need different data -----------
    ("What is the SLA target for approval?",
     "What is the SLA target for invoicing?",
     False, "same domain, different stage — different numbers"),
    ("How many POs are late this week?",
     "How many POs were late last month?",
     False, "same metric, different time window — different numbers"),
    ("What is Supplier A's on-time rate?",
     "What is Supplier B's on-time rate?",
     False, "same metric, different entity — different numbers"),
    ("What is the cycle time for IT?",
     "What is the cycle time for Facilities?",
     False, "same metric, different category — different numbers"),
    ("What does the approval policy say?",
     "What does the escalation policy say?",
     False, "policy lookup, different document section"),
    ("Is the pipeline up to date?",
     "Are there any SLA breaches right now?",
     False, "different tools: get_pipeline_status vs get_sla_metrics"),
    ("What caused the SLA breach for order 4?",
     "What caused the SLA breach for order 12?",
     False, "same question shape, different order ID — different root cause"),
    ("What is the overall conformance rate?",
     "What is the conformance rate for IT orders?",
     False, "overall vs segment — numbers differ"),
    ("How many cases are in the approval stage?",
     "How many cases are in the invoicing stage?",
     False, "same metric, different pipeline stage"),
    ("What is the average cycle time?",
     "What is the 90th-percentile cycle time?",
     False, "same dimension, different aggregation"),
    ("Which supplier has the worst SLA performance?",
     "Which category has the worst SLA performance?",
     False, "same question, different dimension (supplier vs category)"),
    ("Can I raise an emergency purchase order?",
     "What is the threshold for emergency purchases?",
     False, "policy lookup — different question intent despite similar words"),
]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b)) / denom


def main() -> None:
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    all_texts = []
    for anchor, probe, _, _ in PAIRS:
        all_texts.extend([anchor, probe])

    print(f"Embedding {len(all_texts)} texts …")
    embeddings = model.encode(all_texts, convert_to_numpy=True)

    results = []
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    print(f"\n{'Anchor':<50} {'Probe':<50}  {'Sim':>6}  {'Hit?':>5}  {'Correct?':>8}  Rationale")
    print("-" * 140)

    for i, (anchor, probe, same_expected, rationale) in enumerate(PAIRS):
        a_emb = embeddings[i * 2]
        p_emb = embeddings[i * 2 + 1]
        sim = _cosine(a_emb, p_emb)
        cache_hit = sim >= THRESHOLD

        if same_expected and cache_hit:
            outcome = "TP"
            true_positives += 1
        elif not same_expected and cache_hit:
            outcome = "FP"  # false hit
            false_positives += 1
        elif same_expected and not cache_hit:
            outcome = "FN"
            false_negatives += 1
        else:
            outcome = "TN"
            true_negatives += 1

        correct = outcome in ("TP", "TN")
        print(f"{anchor[:48]:<50} {probe[:48]:<50}  {sim:>6.3f}  {'HIT' if cache_hit else 'miss':>5}  "
              f"{'OK' if correct else 'WRONG':>8}  {rationale}")

        results.append({
            "anchor": anchor,
            "probe": probe,
            "cosine_similarity": round(float(sim), 4),
            "cache_hit": cache_hit,
            "expected_same_answer": same_expected,
            "outcome": outcome,
            "rationale": rationale,
        })

    total = len(PAIRS)
    total_hits = true_positives + false_positives
    total_paraphrases = sum(1 for r in results if r["expected_same_answer"])
    total_different = total - total_paraphrases

    fhr = false_positives / total_different if total_different > 0 else 0.0
    paraphrase_hit_rate = true_positives / total_paraphrases if total_paraphrases > 0 else 0.0

    summary = {
        "threshold": THRESHOLD,
        "model": MODEL_NAME,
        "total_pairs": total,
        "paraphrase_pairs": total_paraphrases,
        "different_intent_pairs": total_different,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "false_hit_rate": round(fhr, 4),
        "paraphrase_recall": round(paraphrase_hit_rate, 4),
        "pairs": results,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 80)
    print(f"Threshold:            {THRESHOLD}")
    print(f"Total pairs:          {total}  ({total_paraphrases} paraphrases, {total_different} different-intent)")
    print(f"Cache hits:           {total_hits}")
    print(f"  True positives:     {true_positives}  (paraphrase hits — correct)")
    print(f"  False positives:    {false_positives}  (different-intent hits — wrong)")
    print(f"False-hit rate:       {fhr:.1%}  ({false_positives}/{total_different} different-intent pairs)")
    print(f"Paraphrase recall:    {paraphrase_hit_rate:.1%}  ({true_positives}/{total_paraphrases} paraphrases hit)")
    print(f"\nResults written to: {OUT_PATH}")


if __name__ == "__main__":
    main()
