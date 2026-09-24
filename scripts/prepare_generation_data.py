"""
Build instruction-tuning data for generation fine-tuning.

Calls search_policy_documents (the same retrieval stack the live assistant uses) to
retrieve the most relevant policy chunk for each in-scope question in eval_dataset.json.
No LLM calls needed -- the answer is constructed directly from retrieved chunk text,
keeping the data fully grounded in the actual policy corpus.

Data format: ShareGPT (messages list) per line in JSONL.
Split: 80% train / 20% test, stratified by question category so every category has
test coverage even with only ~130 examples.

Output:
  data/finetune/train.jsonl   -- ShareGPT examples for QLoRA training
  data/finetune/test.jsonl    -- held-out examples for evaluation
  data/finetune/stats.json    -- split sizes, per-category counts, skipped questions

Run: python -m scripts.prepare_generation_data
"""
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
EVAL_PATH = ROOT / "data/evaluation/eval_dataset.json"
OUT_DIR = ROOT / "data/finetune"
SYSTEM_PROMPT = (
    "You are an operations assistant for Northstar Manufacturing's procurement process. "
    "Answer questions about procurement policy, SLA targets, escalation procedures, "
    "vendor onboarding, and contract renewal using only the policy information you know. "
    "Be specific and cite the relevant policy document and section."
)
TRAIN_FRACTION = 0.80
RANDOM_SEED = 42


def _build_answer(citation: str, text: str) -> str:
    """Format a retrieved chunk into a natural assistant-style answer."""
    import re
    # Strip markdown heading markers and table pipes for cleaner prose output
    clean = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)
    clean = clean.strip()
    return f"According to {citation}: {clean}"


def _retrieve_top_chunk(question: str) -> dict | None:
    """Return the top search result for a question, or None if retrieval fails."""
    try:
        from src.tools.registry import call_tool
        result = call_tool("search_policy_documents", query=question)
        if result.get("found") and result.get("results"):
            return result["results"][0]
    except Exception as exc:
        print(f"  [WARN] retrieval failed for {question!r}: {exc}", file=sys.stderr)
    return None


def _stratified_split(items_by_category: dict[str, list], train_frac: float, seed: int):
    """Split per-category lists into (train, test) preserving category ratios."""
    rng = random.Random(seed)
    train, test = [], []
    for cat, items in sorted(items_by_category.items()):
        shuffled = list(items)
        rng.shuffle(shuffled)
        n_train = max(1, math.floor(len(shuffled) * train_frac))
        train.extend(shuffled[:n_train])
        test.extend(shuffled[n_train:])
    return train, test


def build_dataset() -> dict:
    data = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    in_scope = [q for q in data if q.get("in_scope")]

    by_category: dict[str, list] = defaultdict(list)
    skipped = []

    print(f"Processing {len(in_scope)} in-scope questions ...", flush=True)
    for i, q in enumerate(in_scope, 1):
        question = q["question"]
        chunk = _retrieve_top_chunk(question)
        if chunk is None:
            skipped.append({"id": q.get("id"), "question": question, "reason": "retrieval_failed"})
            continue

        answer = _build_answer(chunk["citation"], chunk["text"])
        example = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            "meta": {
                "id": q.get("id"),
                "category": q.get("category"),
                "citation": chunk["citation"],
                "similarity_score": chunk.get("similarity_score"),
            },
        }
        by_category[q.get("category", "unknown")].append(example)
        if i % 20 == 0:
            print(f"  {i}/{len(in_scope)} done", flush=True)

    train, test = _stratified_split(by_category, TRAIN_FRACTION, RANDOM_SEED)

    stats = {
        "n_in_scope": len(in_scope),
        "n_train": len(train),
        "n_test": len(test),
        "n_skipped": len(skipped),
        "train_frac": TRAIN_FRACTION,
        "by_category": {
            cat: {"n_train": sum(1 for e in train if e["meta"]["category"] == cat),
                  "n_test": sum(1 for e in test if e["meta"]["category"] == cat)}
            for cat in sorted(by_category)
        },
        "skipped": skipped,
    }
    return {"train": train, "test": test, "stats": stats}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = build_dataset()

    train_path = OUT_DIR / "train.jsonl"
    test_path = OUT_DIR / "test.jsonl"
    stats_path = OUT_DIR / "stats.json"

    train_path.write_text(
        "\n".join(json.dumps(e) for e in result["train"]) + "\n", encoding="utf-8"
    )
    test_path.write_text(
        "\n".join(json.dumps(e) for e in result["test"]) + "\n", encoding="utf-8"
    )
    stats_path.write_text(json.dumps(result["stats"], indent=2), encoding="utf-8")

    s = result["stats"]
    print(f"\nDataset built:")
    print(f"  train: {s['n_train']} examples")
    print(f"  test:  {s['n_test']} examples")
    print(f"  skipped: {s['n_skipped']}")
    print(f"  per-category:")
    for cat, counts in s["by_category"].items():
        print(f"    {cat:<25} train={counts['n_train']:>3}  test={counts['n_test']:>3}")
    print(f"\nWrote {train_path.relative_to(ROOT)}")
    print(f"      {test_path.relative_to(ROOT)}")
    print(f"      {stats_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
