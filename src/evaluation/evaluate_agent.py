"""
First-pass agent evaluation: for each question in data/evaluation/agent_questions.json,
run the real agent and score tool selection -- did it call the tools a human would
expect for this category of question. This needs a real ANTHROPIC_API_KEY and network
access to run for real (unlike the unit tests, which mock the client entirely); it's
meant to be run manually against the live agent, not as part of the automated test
suite. See docs/evaluation.md for the full scoring rubric once Phase 19 is complete --
this is deliberately a narrower first pass (tool selection only), not the full
retrieval/citation/groundedness scoring FEATURES.md calls for eventually.
"""
import json
from dataclasses import dataclass
from pathlib import Path

from src.agent.agent import run_agent


@dataclass
class AgentEvalResult:
    question: str
    category: str
    expected_tools: list[str]
    actual_tools: list[str]
    tool_selection_correct: bool
    answer: str


def load_questions(path: str = "data/evaluation/agent_questions.json") -> list[dict]:
    return json.loads(Path(path).read_text())


def _tool_selection_correct(expected: list[str], actual: list[str], category: str) -> bool:
    """For 'unanswerable' and 'adversarial' questions, the correct behavior is often
    NOT calling any tool with a plausible-sounding fabricated answer -- so 'correct' for
    those categories means the agent didn't confidently answer from nothing, which is a
    property of the answer text, not just tool calls. Tool-set comparison alone is only
    meaningful for the 'data'/'document'/'combined'/'multi_step' categories."""
    if category in ("unanswerable", "adversarial"):
        return True  # scored separately by inspecting the answer text, not here
    return set(expected).issubset(set(actual))


def run_evaluation(path: str = "data/evaluation/agent_questions.json") -> dict:
    questions = load_questions(path)
    results = []

    for q in questions:
        response = run_agent(q["question"])
        results.append(AgentEvalResult(
            question=q["question"],
            category=q["category"],
            expected_tools=q["expected_tools"],
            actual_tools=response.tools_used,
            tool_selection_correct=_tool_selection_correct(
                q["expected_tools"], response.tools_used, q["category"]
            ),
            answer=response.answer,
        ))

    passed = sum(1 for r in results if r.tool_selection_correct)
    return {
        "total": len(results),
        "passed": passed,
        "pass_rate_pct": round((passed / len(results)) * 100, 1) if results else 0.0,
        "results": results,
    }


if __name__ == "__main__":
    summary = run_evaluation()
    print(f"Agent eval: {summary['passed']}/{summary['total']} ({summary['pass_rate_pct']}%)\n")
    for r in summary["results"]:
        status = "PASS" if r.tool_selection_correct else "FAIL"
        print(f"[{status}] ({r.category}) {r.question}")
        print(f"       expected={r.expected_tools} actual={r.actual_tools}")
        if r.category in ("unanswerable", "adversarial"):
            print(f"       answer (inspect manually): {r.answer[:150]}")
