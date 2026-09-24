"""
Ragas evaluation against the 32 in-domain faithfulness questions.

Runs Ragas faithfulness, context_precision, context_recall, and
answer_relevancy metrics on (question, answer, contexts) triples built by
re-running retrieval on each question from faithfulness_results.json.

Outputs:
  data/evaluation/ragas_results.json   — per-question Ragas scores + NLI score
  (also prints a summary comparison table to stdout)

Usage:
  python -m scripts.evaluate_ragas [--top-k 5] [--model MODEL] [--out PATH] [--resume]

The script re-runs reranked_search (ChromaDB + BM25 + cross-encoder) at
runtime so the contexts reflect the current index state.  Ragas metrics
require an LLM judge; defaults to Groq qwen/qwen3.8-27b via GROQ_API_KEY.
Override with --model or RAGAS_MODEL env var.

API note (ragas 0.4.3): uses the new ragas.metrics.collections API which
calls metric.score(user_input, response, retrieved_contexts) directly per
sample (not the older evaluate()/EvaluationDataset path).
"""
import argparse
import json
import os
import sys
import time
import types
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ragas 0.4.x tries to import ChatVertexAI from langchain_community, which was
# removed in langchain-community 0.3+. Stub it out so the import chain succeeds
# without requiring the google-cloud-aiplatform dependency.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")
    class _ChatVertexAI:
        pass
    _vertexai_stub.ChatVertexAI = _ChatVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_stub

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "evaluation"
INPUT_FILE = DATA_DIR / "faithfulness_results.json"
DEFAULT_OUT = DATA_DIR / "ragas_results.json"


def _build_llm(model: str):
    """Build a ragas InstructorLLM via Groq (OpenAI-compatible) or Anthropic."""
    try:
        from ragas.llms import llm_factory
    except ImportError:
        sys.exit("ragas not installed. Run: pip install ragas")

    groq_key = os.environ.get("GROQ_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if groq_key:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
        return llm_factory(model, provider="openai", client=client), f"groq/{model}"

    if anthropic_key:
        try:
            from anthropic import AsyncAnthropic
            client = AsyncAnthropic(api_key=anthropic_key)
            return llm_factory(model, provider="anthropic", client=client), f"anthropic/{model}"
        except ImportError:
            pass

    sys.exit("Set GROQ_API_KEY or ANTHROPIC_API_KEY and install ragas.")


def _retrieve_contexts(question: str, top_k: int) -> list[str]:
    from src.retrieval.search import reranked_search
    results = reranked_search(question, top_k=top_k)
    return [r.text for r in results]


def _score_question(q: dict, metrics: list, top_k: int) -> dict:
    """Retrieve contexts and run all Ragas metrics for one question."""
    question = q["question"]
    answer = q["answer"]

    contexts = _retrieve_contexts(question, top_k)

    scores = {"n_contexts": len(contexts)}
    for metric in metrics:
        try:
            result = metric.score(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
            )
            scores[f"ragas_{metric.name}"] = float(result.value)
        except Exception as e:
            scores[f"ragas_{metric.name}"] = None
            scores[f"ragas_{metric.name}_error"] = str(e)[:120]

    return scores


def main():
    parser = argparse.ArgumentParser(description="Run Ragas metrics on the 32 in-domain questions")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--model",
        default=os.environ.get("RAGAS_MODEL", "qwen/qwen3.8-27b"),
        help="Model name for Ragas LLM judge (default: qwen/qwen3.8-27b via Groq)",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--resume", action="store_true", help="Skip questions already in output file")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        sys.exit(f"Input file not found: {INPUT_FILE}")
    with open(INPUT_FILE) as f:
        fdata = json.load(f)
    questions = fdata["results"]
    print(f"Loaded {len(questions)} questions from {INPUT_FILE.name}")

    # Load prior results for --resume
    existing: dict[int, dict] = {}
    if args.resume and out_path.exists():
        with open(out_path) as f:
            prior = json.load(f)
        for row in prior.get("results", []):
            existing[row["id"]] = row
        print(f"Resuming — {len(existing)} already scored")

    llm, model_label = _build_llm(args.model)
    print(f"LLM: {model_label}")

    # ragas 0.4.3 collections metrics — score() API, async client required.
    # Faithfulness is the direct counterpart to our NLI gate and needs only an
    # LLM (no embeddings, no reference answer). ContextRecall requires a
    # reference (ground-truth) answer we don't have; AnswerRelevancy requires
    # embeddings. Running faithfulness only keeps the comparison clean.
    try:
        from ragas.metrics.collections import Faithfulness
        metrics = [Faithfulness(llm=llm)]
        print(f"Metrics: {[m.name for m in metrics]}")
    except Exception as e:
        sys.exit(f"Failed to initialise Ragas metrics: {e}")

    results = []
    for i, q in enumerate(questions):
        qid = q["id"]
        if qid in existing:
            results.append(existing[qid])
            continue

        nli_score = q["faithfulness_score"]
        print(f"[{i+1}/{len(questions)}] Q{qid:03d} {q['question'][:55]!r} ...", end=" ", flush=True)
        t0 = time.monotonic()

        try:
            ragas_scores = _score_question(q, metrics, args.top_k)
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "id": qid, "category": q["category"], "question": q["question"],
                "nli_faithfulness": nli_score,
                "error": str(e)[:200],
            })
            _save(results, out_path, model_label, args.top_k)
            continue

        elapsed = round(time.monotonic() - t0, 1)
        row = {
            "id": qid,
            "category": q["category"],
            "question": q["question"],
            "nli_faithfulness": nli_score,
            **ragas_scores,
        }
        results.append(row)

        rf = ragas_scores.get("ragas_faithfulness")
        diff_str = f"{rf - nli_score:+.3f}" if rf is not None else "n/a"
        print(f"NLI={nli_score:.3f}  Ragas={rf if rf is None else f'{rf:.3f}'}  diff={diff_str}  ({elapsed}s)")

        _save(results, out_path, model_label, args.top_k)
        time.sleep(1.5)  # stay under Groq rate limit

    _save(results, out_path, model_label, args.top_k)
    _print_summary(results)
    print(f"\nSaved to {out_path}")


def _save(results: list, out_path: Path, model_label: str, top_k: int):
    valid = [r for r in results if r.get("ragas_faithfulness") is not None]
    out: dict = {
        "model": model_label,
        "top_k": top_k,
        "n_total": len(results),
        "n_scored": len(valid),
        "results": results,
    }
    if valid:
        out["mean_nli_faithfulness"] = round(
            sum(r["nli_faithfulness"] for r in valid) / len(valid), 4
        )
        out["mean_ragas_faithfulness"] = round(
            sum(r["ragas_faithfulness"] for r in valid) / len(valid), 4
        )
        agree = sum(
            1 for r in valid
            if (r["nli_faithfulness"] >= 0.5) == (r["ragas_faithfulness"] >= 0.5)
        )
        out["nli_ragas_direction_agreement_pct"] = round(100 * agree / len(valid), 1)


    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)


def _print_summary(results: list):
    valid = [r for r in results if r.get("ragas_faithfulness") is not None]
    if not valid:
        print("No scored results to summarize.")
        return
    n = len(valid)

    def mean(key: str) -> float:
        vals = [r[key] for r in valid if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else float("nan")

    print(f"\n{'='*62}")
    print(f"Ragas vs NLI gate comparison  (n={n} / {len(results)} total)")
    print(f"{'='*62}")
    print(f"  NLI faithfulness mean:        {mean('nli_faithfulness'):.3f}")
    print(f"  Ragas faithfulness mean:      {mean('ragas_faithfulness'):.3f}")

    agree = sum(
        1 for r in valid
        if (r["nli_faithfulness"] >= 0.5) == (r["ragas_faithfulness"] >= 0.5)
    )
    print(f"\n  NLI / Ragas direction agreement: {100*agree/n:.1f}%")

    diffs = sorted(
        [(abs(r["ragas_faithfulness"] - r["nli_faithfulness"]), r["id"], r) for r in valid],
        reverse=True,
    )
    print(f"\n  Top 5 biggest NLI vs Ragas disagreements:")
    for gap, _, r in diffs[:5]:
        rf = r["ragas_faithfulness"]
        nli = r["nli_faithfulness"]
        print(
            f"    Q{r['id']:03d} [{r['category']:22s}]  "
            f"NLI={nli:.3f}  Ragas={rf:.3f}  diff={rf - nli:+.3f}"
        )


if __name__ == "__main__":
    main()
