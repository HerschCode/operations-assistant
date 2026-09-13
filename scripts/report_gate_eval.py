"""
Read the committed grounded gate results, write reports/p2_gate_eval.json.

Offline -- no GROQ API key needed. Results file is committed:
  data/evaluation/grounded_gate_results.json

To re-run against the live LLM: python scripts/evaluate_grounded_gate.py (requires GROQ_API_KEY).

If reports/p2_retrieval_eval.json exists (produced by scripts/evaluate_retrieval.py),
a compact retrieval metrics section is appended to the report and printed.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_retrieval_summary(retrieval_path: Path) -> dict | None:
    """Read the retrieval eval report and return a compact summary, or None."""
    if not retrieval_path.exists():
        return None
    try:
        data = json.loads(retrieval_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    metrics = data.get("metrics", {})
    summary: dict[str, dict] = {}
    for strategy, strategy_data in metrics.items():
        overall = strategy_data.get("overall", {})
        summary[strategy] = {
            "hit_at_1": overall.get("hit_at_1"),
            "hit_at_3": overall.get("hit_at_3"),
            "hit_at_5": overall.get("hit_at_5"),
            "mrr": overall.get("mrr"),
            "n": overall.get("n"),
        }
    return {
        "top_k": data.get("top_k"),
        "strategies": summary,
    }


def main():
    src = REPO_ROOT / "data" / "evaluation" / "grounded_gate_results.json"
    if not src.exists():
        print("Eval results not found. Run: python scripts/evaluate_grounded_gate.py  (requires GROQ_API_KEY)")
        sys.exit(1)

    data = json.loads(src.read_text())
    records = data.get("records", [])
    thresholds = data.get("thresholds_evaluated", [0.3, 0.5, 0.7])

    threshold_stats = {}
    for t in thresholds:
        gated = [r for r in records if r["faithfulness_score"] < t]
        passing = [r for r in records if r["faithfulness_score"] >= t]
        avg_faith = (
            round(sum(r["faithfulness_score"] for r in passing) / len(passing), 3)
            if passing else 0.0
        )
        threshold_stats[str(t)] = {
            "gate_rate_pct": round(len(gated) / len(records) * 100, 1) if records else 0,
            "coverage_pct": round(len(passing) / len(records) * 100, 1) if records else 0,
            "avg_faithfulness_passing": avg_faith,
        }

    report = {
        "model": data.get("model"),
        "total_questions": len(records),
        "recommended_threshold": data.get("recommended_threshold"),
        "thresholds": threshold_stats,
    }

    # Optionally include retrieval metrics if the retrieval eval has been run.
    retrieval_path = REPO_ROOT / "reports" / "p2_retrieval_eval.json"
    retrieval_summary = _load_retrieval_summary(retrieval_path)
    if retrieval_summary is not None:
        report["retrieval_metrics"] = retrieval_summary
        print("Retrieval metrics (from p2_retrieval_eval.json):")
        for strategy, m in retrieval_summary.get("strategies", {}).items():
            print(
                f"  {strategy:<10}  Hit@1={m['hit_at_1']}%  Hit@3={m['hit_at_3']}%"
                f"  Hit@5={m['hit_at_5']}%  MRR={m['mrr']}  N={m['n']}"
            )
        print()
    else:
        print(
            "Note: retrieval metrics not included (run scripts/evaluate_retrieval.py"
            " to generate reports/p2_retrieval_eval.json)."
        )
        print()

    out = REPO_ROOT / "reports" / "p2_gate_eval.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
