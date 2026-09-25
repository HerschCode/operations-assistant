"""
Compute Cohen's kappa between human labels and LLM judge scores.

Usage:
    python -m scripts.compute_kappa data/evaluation/agent_labels.json

agent_labels.json format:
    [{"id": "Q001", "human_score": 5}, {"id": "Q002", "human_score": 4}, ...]

Reads LLM judge scores from data/evaluation/agent_eval_v2_results.json.
Requires at least --min-overlap questions with both a human score and a judge score.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS_FILE = ROOT / "data/evaluation/agent_eval_v2_results.json"
MIN_OVERLAP = 20


def _load_judge_scores(results_path: Path) -> dict[str, int]:
    data = json.loads(results_path.read_text(encoding="utf-8"))
    rows = data.get("rows", [])
    return {
        r["id"]: int(r["llm_judge_score"])
        for r in rows
        if r.get("llm_judge_score") is not None and "error" not in r
    }


def _load_human_scores(labels_path: Path) -> dict[str, int]:
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    return {entry["id"]: int(entry["human_score"]) for entry in labels}


def cohen_kappa(rater_a: list[int], rater_b: list[int], n_cats: int = 5) -> float:
    """Cohen's kappa for two equal-length ordinal rating lists (1..n_cats)."""
    n = len(rater_a)
    if n == 0:
        return float("nan")

    observed_agree = sum(a == b for a, b in zip(rater_a, rater_b)) / n

    freq_a = Counter(rater_a)
    freq_b = Counter(rater_b)
    expected_agree = sum(
        (freq_a.get(k, 0) / n) * (freq_b.get(k, 0) / n)
        for k in range(1, n_cats + 1)
    )

    if expected_agree == 1.0:
        return 1.0
    return (observed_agree - expected_agree) / (1.0 - expected_agree)


def weighted_kappa(rater_a: list[int], rater_b: list[int], n_cats: int = 5) -> float:
    """Linear-weighted kappa — penalises larger disagreements more."""
    n = len(rater_a)
    if n == 0:
        return float("nan")

    max_diff = n_cats - 1

    observed_weighted = sum(1.0 - abs(a - b) / max_diff for a, b in zip(rater_a, rater_b)) / n

    freq_a = Counter(rater_a)
    freq_b = Counter(rater_b)
    expected_weighted = sum(
        (freq_a.get(i, 0) / n) * (freq_b.get(j, 0) / n) * (1.0 - abs(i - j) / max_diff)
        for i in range(1, n_cats + 1)
        for j in range(1, n_cats + 1)
    )

    if expected_weighted == 1.0:
        return 1.0
    return (observed_weighted - expected_weighted) / (1.0 - expected_weighted)


def _kappa_label(k: float) -> str:
    if k >= 0.80:
        return "almost perfect"
    if k >= 0.61:
        return "substantial"
    if k >= 0.41:
        return "moderate"
    if k >= 0.21:
        return "fair"
    if k >= 0.0:
        return "slight"
    return "poor"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", help="Path to agent_labels.json")
    ap.add_argument("--results", default=str(RESULTS_FILE),
                    help="Path to agent_eval_v2_results.json")
    ap.add_argument("--min-overlap", type=int, default=MIN_OVERLAP,
                    help="Minimum matched questions required (default: 20)")
    args = ap.parse_args()

    results_path = Path(args.results)
    labels_path = Path(args.labels)

    if not results_path.exists():
        print(f"ERROR: results file not found: {results_path}", file=sys.stderr)
        print("Run the agent eval first: python -m scripts.evaluate_agent_v2 --resume", file=sys.stderr)
        sys.exit(1)
    if not labels_path.exists():
        print(f"ERROR: labels file not found: {labels_path}", file=sys.stderr)
        sys.exit(1)

    judge_scores = _load_judge_scores(results_path)
    human_scores = _load_human_scores(labels_path)

    overlap_ids = sorted(set(judge_scores) & set(human_scores))
    if len(overlap_ids) < args.min_overlap:
        print(f"ERROR: only {len(overlap_ids)} questions have both human and judge scores "
              f"(need {args.min_overlap}).", file=sys.stderr)
        print(f"  Judge scored: {len(judge_scores)} questions", file=sys.stderr)
        print(f"  Human scored: {len(human_scores)} questions", file=sys.stderr)
        sys.exit(1)

    human = [human_scores[q] for q in overlap_ids]
    judge = [judge_scores[q] for q in overlap_ids]

    kappa = cohen_kappa(human, judge)
    wkappa = weighted_kappa(human, judge)
    exact_agree = sum(h == j for h, j in zip(human, judge)) / len(human)
    within1 = sum(abs(h - j) <= 1 for h, j in zip(human, judge)) / len(human)

    mean_diff = sum(h - j for h, j in zip(human, judge)) / len(human)

    print(f"\nHuman-Judge Agreement  (n={len(overlap_ids)} questions)")
    print(f"  Cohen's kappa (unweighted): {kappa:.3f}  [{_kappa_label(kappa)}]")
    print(f"  Linear-weighted kappa:      {wkappa:.3f}  [{_kappa_label(wkappa)}]")
    print(f"  Exact agreement:            {exact_agree:.1%}")
    print(f"  Within-1 agreement:         {within1:.1%}")
    print(f"  Mean human - judge bias:    {mean_diff:+.2f}")

    # Per-category breakdown (if category info available)
    results_data = json.loads(results_path.read_text(encoding="utf-8"))
    cat_by_id = {r["id"]: r.get("category", "unknown") for r in results_data.get("rows", [])}
    by_cat: dict[str, tuple[list[int], list[int]]] = {}
    for qid in overlap_ids:
        cat = cat_by_id.get(qid, "unknown")
        h_list, j_list = by_cat.setdefault(cat, ([], []))
        h_list.append(human_scores[qid])
        j_list.append(judge_scores[qid])

    if any(len(v[0]) >= 5 for v in by_cat.values()):
        print("\nPer-category (kappa, n>=5 only):")
        for cat, (h_list, j_list) in sorted(by_cat.items()):
            if len(h_list) < 5:
                continue
            k = cohen_kappa(h_list, j_list)
            print(f"  {cat:<25} kappa={k:.3f}  [{_kappa_label(k)}]  n={len(h_list)}")

    output = {
        "n_overlap": len(overlap_ids),
        "cohen_kappa": round(kappa, 4),
        "weighted_kappa": round(wkappa, 4),
        "exact_agreement": round(exact_agree, 4),
        "within1_agreement": round(within1, 4),
        "mean_human_minus_judge": round(mean_diff, 3),
        "ids": overlap_ids,
    }
    out_path = labels_path.parent / "kappa_results.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    try:
        display = out_path.relative_to(ROOT)
    except ValueError:
        display = out_path
    print(f"\nWrote {display}")


if __name__ == "__main__":
    main()
