"""
Evaluate the faithfulness gate: before/after comparison showing gated vs ungated RAG.

Uses the same 32 in-scope questions as evaluate_faithfulness.py but runs the
full grounded_answer() pipeline at multiple gate thresholds, reporting:
  - Faithfulness score per category (same as before)
  - Gate activation rate (fraction of questions where the gate fired)
  - Answer coverage (fraction of questions that got a real answer, not the fallback)
  - The precision/coverage tradeoff at each threshold

The key question: does the gate actually reduce unfaithful answers without
gating too many legitimate questions?

Run from repo root:
    python scripts/evaluate_grounded_gate.py

Requires GROQ_API_KEY in environment.
"""
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
load_dotenv()

EVAL_PATH = REPO_ROOT / "data/evaluation/eval_dataset.json"
GROQ_MODEL = "openai/gpt-oss-120b"
CONTEXT_TEMPLATE = """\
Answer the question using ONLY the provided context. If the context does not \
contain enough information to answer, say so — do not guess.

Context:
{context}

Question: {question}
"""
GATE_THRESHOLDS = [0.3, 0.5, 0.7]


def _ask_groq(question: str, chunk_texts: list[str]) -> str:
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    context = "\n\n---\n\n".join(chunk_texts)
    prompt = CONTEXT_TEMPLATE.format(context=context, question=question)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def main():
    from src.retrieval.search import reranked_search
    from src.evaluation.faithfulness import score_faithfulness

    dataset = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    in_scope = [q for q in dataset if q["in_scope"]]

    print(f"\n{'='*72}")
    print(f"  Grounded Gate Evaluation — {len(in_scope)} questions, {GROQ_MODEL}")
    print(f"  Gate thresholds: {GATE_THRESHOLDS}")
    print(f"{'='*72}")

    # Run retrieval + LLM once, then re-score at each threshold
    print("\n  Collecting answers (retrieval + LLM)...")
    records = []
    for q in in_scope:
        chunks = reranked_search(q["question"], top_k=5)
        chunk_texts = [r.text for r in chunks]
        answer = _ask_groq(q["question"], chunk_texts)
        result = score_faithfulness(answer, chunk_texts)
        records.append({
            "id": q["id"],
            "category": q["category"],
            "faithfulness_score": result.faithfulness_score,
            "n_sentences": result.n_sentences,
            "n_grounded": result.n_grounded,
        })
        time.sleep(0.3)

    # Report per-threshold precision/coverage tradeoff
    print(f"\n  {'Threshold':>9}  {'Gate rate':>9}  {'Coverage':>9}  {'Avg faith (ungated)':>20}  {'Avg faith (all)':>16}")
    print(f"  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*20}  {'-'*16}")

    for thresh in GATE_THRESHOLDS:
        gated = [r for r in records if r["faithfulness_score"] < thresh]
        passed = [r for r in records if r["faithfulness_score"] >= thresh]
        gate_rate = len(gated) / len(records)
        coverage = len(passed) / len(records)
        avg_faith_passed = sum(r["faithfulness_score"] for r in passed) / len(passed) if passed else 0.0
        avg_faith_all = sum(r["faithfulness_score"] for r in records) / len(records)
        print(f"  {thresh:>9.1f}  {gate_rate:>9.1%}  {coverage:>9.1%}  {avg_faith_passed:>20.1%}  {avg_faith_all:>16.1%}")

    # Per-category at the recommended threshold (0.5)
    recommended = 0.5
    print(f"\n  Per-category at gate_threshold={recommended}:")
    print(f"  {'Category':<22}  {'Answers':>7}  {'Gated':>6}  {'Avg faith':>9}")
    print(f"  {'-'*22}  {'-'*7}  {'-'*6}  {'-'*9}")
    cats = sorted({r["category"] for r in records})
    for cat in cats:
        cat_rows = [r for r in records if r["category"] == cat]
        gated_n = sum(1 for r in cat_rows if r["faithfulness_score"] < recommended)
        avg = sum(r["faithfulness_score"] for r in cat_rows) / len(cat_rows)
        print(f"  {cat:<22}  {len(cat_rows):>7}  {gated_n:>6}  {avg:>9.1%}")

    # Save results
    out = REPO_ROOT / "data/evaluation/grounded_gate_results.json"
    out.write_text(json.dumps({
        "model": GROQ_MODEL,
        "thresholds_evaluated": GATE_THRESHOLDS,
        "recommended_threshold": recommended,
        "records": records,
    }, indent=2), encoding="utf-8")
    print(f"\n  Results saved to {out.relative_to(REPO_ROOT)}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
