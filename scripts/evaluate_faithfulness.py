"""
End-to-end faithfulness evaluation: retrieval → LLM answer → NLI faithfulness score.

For each in-scope question in the eval dataset:
  1. Retrieve top-5 chunks via reranked_search (hybrid + cross-encoder)
  2. Prompt the LLM (Groq) with the retrieved context to generate an answer
  3. Score the answer's faithfulness against the retrieved chunks via NLI

Outputs per-question scores and aggregate stats so you can identify which
question categories tend to produce unfaithful answers.

Run from repo root:
    python scripts/evaluate_faithfulness.py

Requires GROQ_API_KEY in .env (or environment). Uses the same model as the
live agent (config/providers.yaml).
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

from src.retrieval.search import reranked_search
from src.evaluation.faithfulness import score_faithfulness

EVAL_PATH = REPO_ROOT / "data/evaluation/eval_dataset.json"
GROQ_MODEL = "openai/gpt-oss-120b"
CONTEXT_TEMPLATE = """\
Answer the question using ONLY the provided context. If the context does not \
contain enough information to answer, say so — do not guess or use outside knowledge.

Context:
{context}

Question: {question}
"""


def _ask_groq(question: str, context_chunks: list[str]) -> str:
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    context = "\n\n---\n\n".join(context_chunks)
    prompt = CONTEXT_TEMPLATE.format(context=context, question=question)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def main():
    dataset = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    in_scope = [q for q in dataset if q["in_scope"]]

    print(f"\n{'='*72}")
    print(f"  Faithfulness Evaluation — {len(in_scope)} in-scope questions")
    print(f"  Retrieval: Hybrid + Cross-encoder rerank (top-5)")
    print(f"  LLM: {GROQ_MODEL}")
    print(f"  Faithfulness: NLI (cross-encoder/nli-deberta-v3-small)")
    print(f"{'='*72}\n")
    print(f"  {'Category':<22} {'Q':<4} {'Faithful':>8} {'Contradict':>10} {'Score':>7}")
    print(f"  {'-'*22} {'-'*4} {'-'*8} {'-'*10} {'-'*7}")

    all_results = []
    category_stats: dict[str, list] = {}

    for q in in_scope:
        chunks = reranked_search(q["question"], top_k=5)
        chunk_texts = [r.text for r in chunks]

        answer = _ask_groq(q["question"], chunk_texts)
        result = score_faithfulness(answer, chunk_texts)

        entry = {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "answer": answer,
            "faithfulness_score": result.faithfulness_score,
            "contradiction_rate": result.contradiction_rate,
            "n_sentences": result.n_sentences,
            "n_grounded": result.n_grounded,
            "n_contradicted": result.n_contradicted,
            "sentence_scores": [
                {
                    "sentence": s.sentence,
                    "max_entailment": s.max_entailment,
                    "max_contradiction": s.max_contradiction,
                    "grounded": s.grounded,
                    "contradicted": s.contradicted,
                }
                for s in result.sentence_scores
            ],
        }
        all_results.append(entry)

        cat = q["category"]
        category_stats.setdefault(cat, []).append(result.faithfulness_score)

        flag = "✓" if result.faithfulness_score >= 0.8 else "✗"
        print(f"  {cat:<22} {q['id']:<4} {flag} {result.faithfulness_score:>6.0%}  "
              f"{result.contradiction_rate:>9.0%}  {result.n_grounded}/{result.n_sentences}")

        time.sleep(0.3)  # stay within Groq free-tier rate limit

    print(f"\n  {'Category':<22} {'Avg faithfulness':>16} {'n':>4}")
    print(f"  {'-'*22} {'-'*16} {'-'*4}")
    for cat, scores in sorted(category_stats.items()):
        avg = sum(scores) / len(scores)
        print(f"  {cat:<22} {avg:>16.1%} {len(scores):>4}")

    overall = sum(r["faithfulness_score"] for r in all_results) / len(all_results)
    overall_contra = sum(r["contradiction_rate"] for r in all_results) / len(all_results)
    print(f"\n  Overall faithfulness: {overall:.1%}  |  Contradiction rate: {overall_contra:.1%}")
    print(f"{'='*72}\n")

    # Save full results for further analysis
    out = REPO_ROOT / "data/evaluation/faithfulness_results.json"
    out.write_text(json.dumps({"overall_faithfulness": overall,
                               "overall_contradiction_rate": overall_contra,
                               "model": GROQ_MODEL,
                               "results": all_results}, indent=2), encoding="utf-8")
    print(f"  Full results saved to {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
