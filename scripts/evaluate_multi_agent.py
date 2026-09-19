"""
Measures whether the independent Reviewer agent actually improves grounding.

For each question in data/evaluation/agent_questions.json the Researcher runs ONCE; the
Reviewer is then applied to that same draft, so "unreviewed" and "reviewed" answers come
from an identical Researcher run (no sampling noise between arms). Both are scored with the
mechanical number-grounding check (src/evaluation/evaluate_answers.check_groundedness):
a number in the answer that appears nowhere in this turn's tool results is "ungrounded".

Reported honestly either way: how often the Reviewer approved / revised / errored, how many
ungrounded numbers the draft vs final answers contain, and any case where the Reviewer made
a grounded answer worse. Citation traceability is not compared because citations are derived
from tool calls, not from answer text, so a revision cannot change them.

Env: AGENT_PROVIDER=groq AGENT_MODEL=openai/gpt-oss-120b GROQ_API_KEY,
     OPS_PERFORMANCE_API_URL / OPS_PERFORMANCE_API_KEY (the P1 API the data tools call).
Run: python -X utf8 -m scripts.evaluate_multi_agent
"""
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

from src.agent.agent import load_agent_config, run_agent
from src.agent.multi_agent import review_answer
from src.evaluation.evaluate_answers import check_groundedness

ROOT = Path(__file__).parent.parent
QUESTIONS = ROOT / "data/evaluation/agent_questions.json"
OUT = ROOT / "data/evaluation/multi_agent_results.json"


def _retry(fn, attempts=4):
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 -- eval harness: back off on transient/rate errors
            if i == attempts - 1:
                raise
            wait = 15 * (i + 1)
            print(f"  [{type(exc).__name__}] retrying in {wait}s")
            time.sleep(wait)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    config = load_agent_config()
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    rows = []
    for i, q in enumerate(questions):
        if i:
            time.sleep(8)
        try:
            draft = _retry(lambda: run_agent(q["question"]))
        except Exception as exc:  # noqa: BLE001
            print(f"[{i+1}] researcher failed: {exc}")
            rows.append({"question": q["question"], "category": q["category"], "researcher_error": str(exc)})
            continue
        review, revised = _retry(lambda: review_answer(q["question"], draft, config))
        final_answer = revised if review.verdict == "revise" and revised else draft.answer
        g_draft = check_groundedness(draft)
        g_final = check_groundedness(replace(draft, answer=final_answer))
        row = {
            "question": q["question"], "category": q["category"], "verdict": review.verdict,
            "revised": final_answer != draft.answer, "issues": review.issues, "review_error": review.error,
            "ungrounded_draft": g_draft.ungrounded_numbers, "ungrounded_final": g_final.ungrounded_numbers,
            "draft_answer": draft.answer, "final_answer": final_answer,
        }
        rows.append(row)
        print(f"[{i+1}/{len(questions)}] ({q['category']}) verdict={review.verdict} "
              f"ungrounded {len(g_draft.ungrounded_numbers)}->{len(g_final.ungrounded_numbers)}")

    ok = [r for r in rows if "researcher_error" not in r]
    verdicts = {v: sum(r["verdict"] == v for r in ok) for v in ("approve", "revise", "error")}
    summary = {
        "n_questions": len(questions), "n_completed": len(ok), "verdicts": verdicts,
        "answers_changed": sum(r["revised"] for r in ok),
        "answers_with_ungrounded_numbers_draft": sum(bool(r["ungrounded_draft"]) for r in ok),
        "answers_with_ungrounded_numbers_final": sum(bool(r["ungrounded_final"]) for r in ok),
        "total_ungrounded_numbers_draft": sum(len(r["ungrounded_draft"]) for r in ok),
        "total_ungrounded_numbers_final": sum(len(r["ungrounded_final"]) for r in ok),
        "made_worse": [r["question"] for r in ok if len(r["ungrounded_final"]) > len(r["ungrounded_draft"])],
    }
    OUT.write_text(json.dumps({"config": {"provider": config.get("provider"), "model": config.get("model")},
                               "summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
