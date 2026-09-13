"""
Read the committed grounded gate results, write reports/p2_gate_eval.json.

Offline -- no GROQ API key needed. Results file is committed:
  data/evaluation/grounded_gate_results.json

To re-run against the live LLM: python scripts/evaluate_grounded_gate.py (requires GROQ_API_KEY).
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


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

    out = REPO_ROOT / "reports" / "p2_gate_eval.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
