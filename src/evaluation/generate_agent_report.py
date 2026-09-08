"""
Phase 29: renders evaluate_agent.py's scoring into the exact table format an agent
correctness audit needs -- Question | Expected tools | Actual tools | Correct? -- and,
critically, an investigation note for every row that scored incorrect. A pass/fail
count ("4/6 passed") tells you almost nothing on its own; the table is what actually
lets you find and fix a systematic tool-selection problem instead of a one-off.
"""
from dataclasses import dataclass

from src.agent.agent import AgentResponse
from src.evaluation.evaluate_agent import AgentEvalResult, run_evaluation
from src.evaluation.evaluate_adversarial import score_unanswerable, score_adversarial


@dataclass
class InvestigatedResult:
    result: AgentEvalResult
    investigation: str | None  # None if the row was correct -- nothing to investigate


def _investigate(result: AgentEvalResult) -> str | None:
    """Automated first-pass triage of WHY a row scored incorrect -- not a substitute
    for a human reading the actual answer text, but enough to sort 'missing an
    expected tool' from 'called an unexpected one' from 'unanswerable/adversarial
    category needing manual answer-text review' without re-deriving that by hand
    for every failure.

    unanswerable/adversarial rows are always tool_selection_correct=True by
    definition (see evaluate_agent.py), so this function's OWN check for them runs
    unconditionally below, not gated behind the tool_selection_correct branch the
    other categories use."""
    if result.category in ("unanswerable", "adversarial"):
        response = AgentResponse(
            answer=result.answer, tool_calls=result.tool_calls,
            tools_used=result.actual_tools, citations=result.citations,
        )
        if result.category == "unanswerable":
            score = score_unanswerable(response)
        else:
            score = score_adversarial(response, compliance_markers=result.compliance_markers)

        if score.passed:
            return None
        return f"Automated adversarial/unanswerable check FAILED: {score.reason}"

    if result.tool_selection_correct:
        return None

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
        # unanswerable/adversarial rows are always tool_selection_correct=True by
        # definition -- their real pass/fail is whether _investigate found a
        # failing automated check (an investigation string present for these two
        # categories now means "the automated check failed", not just "worth a
        # human look" the way it does for the other categories).
        if r.category in ("unanswerable", "adversarial"):
            correct_mark = "FAIL" if row.investigation else "PASS"
        else:
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
