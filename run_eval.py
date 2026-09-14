"""Loads .env then runs agent evaluation — avoids shell env-export issues on Windows."""
import os
from pathlib import Path

env_file = Path(__file__).parent / ".env"
for line in env_file.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

# gpt-oss-20b: token usage per question fits under 8000 TPM limit now that
# get_supplier_performance returns only top 15 results (~375 tokens, not 5754)
os.environ["AGENT_MODEL"] = "openai/gpt-oss-20b"

from src.evaluation.evaluate_agent import run_evaluation
# 30s between questions: Groq's rate-limit window is ~60s, each question
# typically uses 2-4 API calls, so 30s headroom avoids cascading retries.
from src.evaluation.generate_agent_report import render_markdown_table

summary = run_evaluation(inter_question_delay=30.0)
print(f"\nAgent eval: {summary['passed']}/{summary['total']} ({summary['pass_rate_pct']}%)\n")
for r in summary["results"]:
    status = "PASS" if r.tool_selection_correct else "FAIL"
    print(f"[{status}] ({r.category}) {r.question[:60]}")
    if not r.tool_selection_correct:
        print(f"       expected={r.expected_tools} actual={r.actual_tools}")

print("\n\n--- MARKDOWN REPORT ---\n")
print(render_markdown_table(summary))
