"""
OOD abstention evaluation: does the faithfulness gate correctly refuse to
answer questions that are outside the procurement knowledge base?

Uses SQuAD 2.0 "unanswerable" questions (is_impossible=True) from the
validation set as independently authored OOD queries. These questions span
history, science, sports and pop culture — nothing that appears in the 10
procurement policy documents.

The LLM receives an UNCONSTRAINED prompt (no "answer from context only"
instruction). It will answer from parametric knowledge. The faithfulness
gate then checks whether that answer is entailed by the retrieved procurement
chunks. If not, the gate fires and returns INSUFFICIENT_DATA_MSG.

This tests the core claim: the gate catches answers that don't come from
the source documents, not just answers where the LLM admits ignorance.

Run:
    python -X utf8 -m scripts.evaluate_ood_abstention [--n 50] [--provider groq|gemini]

Requires: GROQ_API_KEY (default) or GOOGLE_API_KEY (--provider gemini);
`pip install datasets` (HuggingFace).
Use --provider gemini when the Groq daily quota is exhausted (resets midnight UTC).
Results: data/evaluation/ood_abstention_results.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv()

OUT = REPO_ROOT / "data" / "evaluation" / "ood_abstention_results.json"
GROQ_MODEL = "openai/gpt-oss-120b"
GEMINI_MODEL = "gemini-2.0-flash"


def _save(rows: list[dict], questions: list[dict]) -> None:
    """Write incremental results so a Groq quota kill doesn't lose everything."""
    ok = [r for r in rows if "error" not in r]
    n_gated = sum(r["gated"] for r in ok)
    partial_summary = {
        "dataset": "SQuAD 2.0 validation — unanswerable (is_impossible=True)",
        "n_questions": len(questions),
        "n_evaluated_so_far": len(ok),
        "n_errors_so_far": len(rows) - len(ok),
        "n_gated_so_far": n_gated,
        "abstention_rate_so_far": round(n_gated / len(ok), 4) if ok else 0,
        "gate_threshold": 0.5,
        "status": "in_progress",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"summary": partial_summary, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _load_squad_unanswerable(n: int) -> list[dict]:
    """Return up to n unanswerable questions from SQuAD 2.0 validation set.

    Selects one question per Wikipedia article title so questions span
    multiple topics rather than clustering around one passage.
    """
    import random
    from datasets import load_dataset
    ds = load_dataset("rajpurkar/squad_v2", split="validation", trust_remote_code=False)
    impossible = [ex for ex in ds if ex["answers"]["text"] == []]
    # Group by title, pick one random question per title, shuffle titles
    by_title: dict[str, list[str]] = {}
    for ex in impossible:
        by_title.setdefault(ex["title"], []).append(ex["question"])
    titles = list(by_title.keys())
    random.seed(42)
    random.shuffle(titles)
    sampled: list[dict] = []
    for title in titles:
        q = random.choice(by_title[title])
        sampled.append({"question": q, "context_topic": title})
        if len(sampled) >= n:
            break
    return sampled


def _ask_llm_unconstrained(question: str, provider: str = "groq") -> str:
    """Answer question from parametric knowledge — no retrieval context injected.
    This is intentional: we want to measure whether the faithfulness gate catches
    answers that use outside knowledge, not answers where the LLM says 'I don't know'
    because we told it to stick to context.

    Use provider='gemini' when the Groq daily free-tier quota is exhausted."""
    if provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = genai.GenerativeModel(GEMINI_MODEL)
        resp = model.generate_content(question,
                                      generation_config={"max_output_tokens": 80, "temperature": 0})
        return resp.text.strip()
    # default: groq
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": question}],
        max_tokens=80,  # short factual answers only — saves quota (eval uses ~4k tokens total)
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def _retry(fn, attempts=3):
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if i == attempts - 1:
                raise
            wait = 20 * (i + 1)
            print(f"  [{type(exc).__name__}] retrying in {wait}s")
            time.sleep(wait)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, metavar="N",
                    help="Number of SQuAD unanswerable questions to evaluate (default 50).")
    ap.add_argument("--resume", action="store_true",
                    help="Continue from an existing partial results file (skips already-evaluated questions).")
    ap.add_argument("--provider", choices=["groq", "gemini"], default="groq",
                    help="LLM provider for unconstrained answers (default: groq). "
                         "Use --provider gemini when the Groq daily quota is exhausted (resets midnight UTC).")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")

    from src.retrieval.grounded_search import grounded_answer, INSUFFICIENT_DATA_MSG

    print(f"Loading {args.n} SQuAD 2.0 unanswerable questions…")
    questions = _load_squad_unanswerable(args.n)
    print(f"  Loaded {len(questions)} questions spanning {len({q['context_topic'] for q in questions})} topics.")

    rows: list[dict] = []
    completed: set[str] = set()
    if args.resume and OUT.exists():
        existing = json.loads(OUT.read_text(encoding="utf-8"))
        rows = [r for r in existing.get("rows", []) if "error" not in r]
        completed = {r["question"] for r in rows}
        print(f"  Resuming: {len(completed)} already evaluated, {len(questions) - len(completed)} remaining.")

    for i, q in enumerate(questions):
        if q["question"] in completed:
            print(f"[{i+1}/{len(questions)}] skip (done): {q['question'][:60]}")
            continue
        if rows:
            time.sleep(3 if args.provider == "gemini" else 5)  # rate-limit pause between API calls
        print(f"[{i+1}/{len(questions)}] {q['context_topic']:30} {q['question'][:60]}")

        # Step 1: get an unconstrained LLM answer (uses parametric knowledge)
        try:
            parametric_answer = _retry(lambda: _ask_llm_unconstrained(q["question"], args.provider))
        except Exception as exc:
            print(f"  LLM failed: {exc}")
            rows.append({"question": q["question"], "topic": q["context_topic"],
                         "error": str(exc)})
            continue

        # Step 2: run through the faithfulness gate
        # We bypass grounded_answer's llm_fn and inject the already-generated answer
        # so we can measure the gate independently of LLM behaviour.
        try:
            from src.retrieval.search import reranked_search
            from src.evaluation.faithfulness import score_faithfulness

            chunks = _retry(lambda: reranked_search(q["question"], top_k=5))
            chunk_texts = [r.text for r in chunks]
            fs = score_faithfulness(parametric_answer, chunk_texts)
            gated = fs.faithfulness_score < 0.5
        except Exception as exc:
            print(f"  Gate failed: {exc}")
            rows.append({"question": q["question"], "topic": q["context_topic"],
                         "parametric_answer": parametric_answer, "error": str(exc)})
            continue

        rows.append({
            "question": q["question"],
            "topic": q["context_topic"],
            "parametric_answer": parametric_answer,
            "faithfulness_score": fs.faithfulness_score,
            "gated": gated,
            "n_sentences": fs.n_sentences,
            "n_grounded": fs.n_grounded,
        })
        verdict = "GATED (correct)" if gated else "passed through"
        print(f"  faith={fs.faithfulness_score:.3f}  {verdict}")
        _save(rows, questions)  # incremental — survives quota interruption

    # Summary
    ok = [r for r in rows if "error" not in r]
    n_gated = sum(r["gated"] for r in ok)
    abstention_rate = n_gated / len(ok) if ok else 0
    topic_breakdown: dict[str, dict] = {}
    for r in ok:
        t = r["topic"]
        if t not in topic_breakdown:
            topic_breakdown[t] = {"total": 0, "gated": 0}
        topic_breakdown[t]["total"] += 1
        topic_breakdown[t]["gated"] += r["gated"]

    summary = {
        "dataset": "SQuAD 2.0 validation — unanswerable (is_impossible=True)",
        "n_questions": len(questions),
        "n_evaluated": len(ok),
        "n_errors": len(rows) - len(ok),
        "n_gated": n_gated,
        "abstention_rate": round(abstention_rate, 4),
        "gate_threshold": 0.5,
        "llm_prompt": "unconstrained (no context injected — tests gate, not LLM refusal)",
        "topic_breakdown": topic_breakdown,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
                   encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"Abstention rate: {n_gated}/{len(ok)} = {abstention_rate:.1%}")
    print(f"(gate fired = correctly refused to answer with outside knowledge)")
    print(f"Results saved to {OUT}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
