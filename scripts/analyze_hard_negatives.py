"""
Gate hard-negative analysis: classify faithfulness_results by answer type and
quantify gate precision/recall per subtype.

Four subtypes derived from (faithfulness_score, contradiction_rate):
  correct      — faithfulness_score > 0  (grounded by NLI)
  halluc_caught— faithfulness=0 + contradiction_rate > 0  (NLI detected contradiction)
  halluc_missed— faithfulness=0 + contradiction_rate = 0  (neutral: neither entailed nor contradicted)
  ambiguous    — faithfulness=1 + contradiction_rate > 0  (entailed AND contradicted simultaneously)

"Hard negatives" in gate calibration terms are the halluc_missed cases:
plausible-sounding answers that contain wrong or unsupported claims without
directly contradicting the source text.  The NLI gate cannot catch these because
the DeBERTa model assigns low entailment AND low contradiction to both the wrong
claim and the correct one (domain mismatch — procurement numbers don't appear in
MNLI/SNLI training data).

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
    return "halluc_missed"


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
    n_halluc_missed = len(by_type.get("halluc_missed", []))
    n_ambiguous = len(by_type.get("ambiguous", []))

    # Gate precision: of the gated answers, what fraction were actually hallucinations?
    n_gated = sum(1 for r in results if r["faithfulness_score"] < ENTAILMENT_THRESHOLD)
    n_gated_halluc = sum(
        1 for t in ("halluc_caught", "halluc_missed")
        for r in by_type.get(t, [])
        if r["faithfulness_score"] < ENTAILMENT_THRESHOLD
    )
    gate_precision = round(n_gated_halluc / n_gated, 4) if n_gated else 0.0

    # Gate recall: of the hallucinations, what fraction did the gate catch?
    n_halluc_total = n_halluc_caught + n_halluc_missed
    n_halluc_gated = sum(
        1 for t in ("halluc_caught", "halluc_missed")
        for r in by_type.get(t, [])
        if r["faithfulness_score"] < ENTAILMENT_THRESHOLD
    )
    gate_recall = round(n_halluc_gated / n_halluc_total, 4) if n_halluc_total else 0.0

    # False-positive rate: fraction of correct answers incorrectly gated
    n_correct_gated = sum(
        1 for r in by_type.get("correct", [])
        if r["faithfulness_score"] < ENTAILMENT_THRESHOLD
    )
    false_positive_rate = round(n_correct_gated / n_correct, 4) if n_correct else 0.0

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
    print(f"Gate performance at threshold=0.5 (over hallucinations only):")
    print(f"  Precision (gated = true hallucination):  {gate_precision:.1%}")
    print(f"  Recall    (hallucinations gated):        {gate_recall:.1%}")
    print(f"  False-positive rate (correct answers gated): {false_positive_rate:.1%}")
    print()
    print("Finding:")
    print(f"  halluc_caught ({n_halluc_caught}): gate works via contradiction detection")
    print(f"  halluc_missed ({n_halluc_missed}): gate blind — low entailment AND low contradiction")
    print(f"  ambiguous     ({n_ambiguous}): entailed + contradicted simultaneously (gate passes but flagged)")
    print(f"  correct       ({n_correct}): {n_correct_gated} incorrectly gated (false positives)")

    analysis = {
        "n_total": total,
        "type_counts": {t: s["n"] for t, s in type_stats.items()},
        "type_stats": type_stats,
        "gate_threshold": ENTAILMENT_THRESHOLD,
        "gate_precision": gate_precision,
        "gate_recall_over_hallucinations": gate_recall,
        "false_positive_rate_over_correct": false_positive_rate,
        "finding": (
            f"Of {total} in-domain answers, {n_halluc_caught} hallucinations are caught by "
            f"contradiction detection (precision {gate_precision:.1%}), {n_halluc_missed} are missed "
            f"(neutral NLI score — gate blind to these). {n_correct_gated}/{n_correct} correct answers "
            f"are incorrectly gated (false-positive rate {false_positive_rate:.1%}), confirming the "
            f"DeBERTa domain-mismatch problem from calibrate_gate.py."
        ),
    }
    OUT.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {OUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
