"""
Phase 29: renders evaluate_agent.py's scoring into the exact table format an agent
correctness audit needs -- Question | Expected tools | Actual tools | Correct? -- and,
critically, an investigation note for every row that scored incorrect. A pass/fail
count ("4/6 passed") tells you almost nothing on its own; the table is what actually
lets you find and fix a systematic tool-selection problem instead of a one-off.
"""
from dataclasses import dataclass

from src.evaluation.evaluate_agent import AgentEvalResult, run_evaluation


@dataclass
class InvestigatedResult:
    result: AgentEvalResult
    investigation: str | None  # None if the row was correct -- nothing to investigate


def _investigate(result: AgentEvalResult) -> str | None:
    """Automated first-pass triage of WHY a row scored incorrect -- not a substitute
    for a human reading the actual answer text, but enough to sort 'missing an
    expected tool' from 'called an unexpected one' from 'unanswerable/adversarial
    category needing manual answer-text review' without re-deriving that by hand
    for every failure."""
    if result.tool_selection_correct:
        return None

    if result.category in ("unanswerable", "adversarial"):
        return (
            "Scored by tool selection only, which is always 'correct' for this category "
            "(see evaluate_agent.py) -- the real question is whether the ANSWER TEXT avoided "
            "fabricating a response or resisted the embedded instruction. Read `answer` manually."
        )

    missing = set(result.expected_tools) - set(result.actual_tools)
    extra = set(result.actual_tools) - set(result.expected_tools)

    parts = []
    if missing:
        parts.append(f"missing expected tool(s): {sorted(missing)}")
    if extra:
        parts.append(f"called unexpected tool(s): {sorted(extra)}")
    return "; ".join(parts) if parts else "tool sets differ in a way not captured by subset check"


def build_report_rows(summary: dict) -> list[InvestigatedResult]:
    return [InvestigatedResult(result=r, investigation=_investigate(r)) for r in summary["results"]]


def render_markdown_table(summary: dict) -> str:
    rows = build_report_rows(summary)
    lines = [
        "# Agent Correctness Audit",
        "",
        f"**{summary['passed']}/{summary['total']} ({summary['pass_rate_pct']}%)** tool-selection "
        "checks passed.",
        "",
        "| Question | Category | Expected tools | Actual tools | Correct? | Investigation |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        r = row.result
        correct_mark = "PASS" if r.tool_selection_correct else "FAIL"
        investigation = row.investigation or "--"
        lines.append(
            f"| {r.question} | {r.category} | {', '.join(r.expected_tools) or '(none)'} | "
            f"{', '.join(r.actual_tools) or '(none)'} | {correct_mark} | {investigation} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    summary = run_evaluation()
    print(render_markdown_table(summary))
