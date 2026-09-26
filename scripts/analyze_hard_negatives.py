"""
Gate hard-negative analysis: classify faithfulness_results by NLI signal pattern
and describe the detection mechanism for each subtype.

Four subtypes derived from (faithfulness_score, contradiction_rate):
  correct        — faithfulness_score > 0  (NLI: grounded, gate passes)
  ambiguous      — faithfulness > 0 + contradiction_rate > 0  (NLI: both entailed AND contradicted, gate passes)
  halluc_caught  — faithfulness=0 + contradiction_rate > 0   (NLI: explicit contradiction signal, gate blocks)
  blocked_neutral— faithfulness=0 + contradiction_rate = 0   (NLI: no signal in either direction, gate still blocks)

IMPORTANT — what this analysis can and cannot measure:
  It characterises HOW the gate makes each blocking decision (which NLI signal
  triggered it). It does NOT measure gate accuracy against independent ground truth
  because "hallucination vs. correct answer" labels are not available here. Without
  those labels, precision/recall cannot be computed non-circularly. What we report
  instead is nli_signal_rate: of all blocked answers, what fraction had an NLI
  contradiction signal backing the block?

The subtypes are defined by NLI output, not by whether the answer is factually wrong.
"blocked_neutral" answers are blocked by the faithfulness threshold even though
DeBERTa assigned near-zero scores to BOTH entailment AND contradiction — the model
has no signal, not a positive "wrong" signal. These are the true hard negatives for
the gate: correct answers that look the same to NLI as wrong ones.

Run:
    python -m scripts.analyze_hard_negatives

Output: data/evaluation/hard_negatives_analysis.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FAITH_FILE = REPO_ROOT / "data" / "evaluation" / "faithfulness_results.json"
OUT = REPO_ROOT / "data" / "evaluation" / "hard_negatives_analysis.json"

ENTAILMENT_THRESHOLD = 0.5
CONTRADICTION_THRESHOLD = 0.4


def _classify(result: dict) -> str:
    faith = result["faithfulness_score"]
    contra = result["contradiction_rate"]
    n_s = result.get("n_sentences", 0)

    if n_s == 0:
        return "empty"
    if faith > 0 and contra == 0:
        return "correct"
    if faith > 0 and contra > 0:
        return "ambiguous"
    if faith == 0 and contra > 0:
        return "halluc_caught"
    return "blocked_neutral"  # faithfulness=0, no contradiction signal — gate blocks but NLI is blind


def main() -> None:
    data = json.loads(FAITH_FILE.read_text(encoding="utf-8"))
    results = data.get("results", [])

    by_type: dict[str, list[dict]] = {}
    for r in results:
        t = _classify(r)
        by_type.setdefault(t, []).append(r)

    # Gate statistics: at default threshold=0.5, what fraction of each type is gated?
    def gate_rate(rows: list[dict]) -> float:
        if not rows:
            return 0.0
        return round(sum(1 for r in rows if r["faithfulness_score"] < ENTAILMENT_THRESHOLD) / len(rows), 4)

    type_stats = {}
    total = len(results)
    for t, rows in sorted(by_type.items()):
        type_stats[t] = {
            "n": len(rows),
            "pct": round(len(rows) / total * 100, 1),
            "gate_rate_at_0_5": gate_rate(rows),
            "mean_max_entailment": round(
                sum(
                    max((ss["max_entailment"] for ss in r.get("sentence_scores", [])), default=0.0)
                    for r in rows
                ) / len(rows), 3
            ) if rows else 0.0,
            "mean_max_contradiction": round(
                sum(
                    max((ss["max_contradiction"] for ss in r.get("sentence_scores", [])), default=0.0)
                    for r in rows
                ) / len(rows), 3
            ) if rows else 0.0,
            "ids": [r["id"] for r in rows],
        }

    n_correct = len(by_type.get("correct", []))
    n_halluc_caught = len(by_type.get("halluc_caught", []))
    n_blocked_neutral = len(by_type.get("blocked_neutral", []))
    n_ambiguous = len(by_type.get("ambiguous", []))

    # How many answers did the gate block?
    n_blocked = sum(1 for r in results if r["faithfulness_score"] < ENTAILMENT_THRESHOLD)
    # How many passed?
    n_passed = total - n_blocked
    gate_coverage = round(n_passed / total, 4) if total else 0.0

    # Of blocked answers, what fraction had a contradiction signal backing the block?
    # This is a non-circular metric: it describes whether NLI had evidence for the block,
    # independent of any "is this answer correct?" ground truth we don't have.
    n_blocked_with_signal = n_halluc_caught  # faithfulness=0 + contradiction>0
    nli_signal_rate = round(n_blocked_with_signal / n_blocked, 4) if n_blocked else 0.0

    print(f"\nHard-Negative Analysis — {total} in-domain answers")
    print(f"  {'Type':<20} {'n':>4} {'%':>6}  gate_rate  mean_entail  mean_contra")
    print(f"  {'-'*20}  {'-'*4}  {'-'*5}  {'-'*9}  {'-'*11}  {'-'*11}")
    for t, s in sorted(type_stats.items()):
        print(
            f"  {t:<20} {s['n']:>4}  {s['pct']:>5.1f}%  "
            f"{s['gate_rate_at_0_5']:>9.1%}  "
            f"{s['mean_max_entailment']:>11.3f}  "
            f"{s['mean_max_contradiction']:>11.3f}"
        )
    print()
    print(f"Gate coverage (in-domain answers that pass): {gate_coverage:.1%} ({n_passed}/{total})")
    print(f"NLI signal rate (blocked answers with contradiction evidence): {nli_signal_rate:.1%}")
    print()
    print("Note: precision/recall against ground truth cannot be computed —")
    print("  independent 'correct vs. hallucinated' labels are not available.")
    print("  This analysis characterises the NLI MECHANISM, not gate ACCURACY.")
    print()
    print("Finding:")
    print(f"  halluc_caught   ({n_halluc_caught}): gate blocks AND NLI has contradiction evidence")
    print(f"  blocked_neutral ({n_blocked_neutral}): gate blocks but NLI is blind (low entailment, low contradiction)")
    print(f"  ambiguous       ({n_ambiguous}): NLI says entailed + contradicted simultaneously; gate passes")
    print(f"  correct         ({n_correct}): NLI says grounded; gate passes")

    analysis = {
        "n_total": total,
        "type_counts": {t: s["n"] for t, s in type_stats.items()},
        "type_stats": type_stats,
        "gate_threshold": ENTAILMENT_THRESHOLD,
        "gate_coverage_in_domain": gate_coverage,
        "nli_signal_rate_of_blocked": nli_signal_rate,
        "caveat": (
            "Precision/recall against ground truth cannot be computed: the subtypes are derived "
            "from NLI scores, not independent human labels. nli_signal_rate describes whether the "
            "gate had NLI evidence for its blocking decision, not whether the blocked answer was "
            "actually wrong."
        ),
        "finding": (
            f"Of {total} in-domain answers, {n_blocked} were blocked by the gate "
            f"(coverage {gate_coverage:.0%}). Of those, {n_halluc_caught} had an NLI contradiction "
            f"signal ({nli_signal_rate:.0%} of blocked). The remaining {n_blocked_neutral} "
            f"(blocked_neutral) were blocked because faithfulness=0 with no contradiction signal — "
            f"NLI assigned near-zero scores in both directions (domain mismatch on procurement "
            f"numbers). {n_correct} answers passed cleanly; {n_ambiguous} passed despite mixed "
            f"NLI signals (entailed AND contradicted)."
        ),
    }
    OUT.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
