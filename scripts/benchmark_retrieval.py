"""
Head-to-head retrieval benchmark: BM25-only vs semantic-only vs hybrid.

Runs all methods against the 120-question eval dataset
(data/evaluation/eval_dataset.json) and reports:

  - Hit@1, Hit@3, Hit@5, MRR   — for the 91 in-scope questions
  - False-positive rate          — for the 29 OOD/adversarial questions
  - Per-category Hit@3 breakdown (hybrid method)
  - Per-method latency

A "hit" for an in-scope question: at least one of the top-k results matches the
expected document ID AND the expected section title (case-insensitive contains).

Run from the project root after indexing documents:
    python -m scripts.index_documents   # if not done
    python scripts/benchmark_retrieval.py

The MIN_SIMILARITY filter applies only to semantic and hybrid search
(BM25 scores are not on a [0,1] scale, so filtering by it would be meaningless).
Out-of-domain questions use a lower threshold (0.1) to be conservative about what
counts as a "retrieval" — if the method returns anything with score > 0.1, that
is counted as a false positive for OOD/adversarial questions.
"""
import json
import time
from pathlib import Path

from src.retrieval.search import bm25_search, semantic_search, hybrid_search, reranked_search

EVAL_PATH = Path(__file__).parent.parent / "data/evaluation/eval_dataset.json"
TOP_K = 5
SEM_MIN_SIMILARITY = 0.3   # same as DEFAULT_MIN_SIMILARITY in search.py
OOD_FP_THRESHOLD = 0.15    # if semantic score > this on an OOD question it's a false positive


def _is_hit(results, expected_doc_id: str, expected_section: str) -> bool:
    """True if any result matches both the expected document and contains the expected
    section substring in its section_title. Section matching is case-insensitive
    substring so minor phrasing differences don't cause spurious misses."""
    for r in results:
        if r.document_id != expected_doc_id:
            continue
        if expected_section is None:
            return True
        section = r.section_title or ""
        if expected_section.lower() in section.lower():
            return True
    return False


def _mrr(results_list: list[list], expected_docs: list[str], expected_sections: list[str]) -> float:
    total = 0.0
    for results, doc_id, section in zip(results_list, expected_docs, expected_sections):
        for rank, r in enumerate(results, start=1):
            if r.document_id == doc_id:
                section_match = section is None or (section.lower() in (r.section_title or "").lower())
                if section_match:
                    total += 1.0 / rank
                    break
    return total / len(results_list) if results_list else 0.0


def _is_ood_false_positive(results) -> bool:
    """For OOD/adversarial questions: a false positive is any result returned.
    BM25 always returns results (no similarity threshold), so for BM25 we check
    if the top result's BM25 score is meaningfully nonzero (> 0.5 normalized).
    For semantic/hybrid, any result above OOD_FP_THRESHOLD counts."""
    if not results:
        return False
    top = results[0]
    # BM25 results have raw BM25 scores, not [0,1] — use a relative check
    # (top score > 0 with content means BM25 "found" something)
    return True   # BM25/semantic/hybrid all returned a result — caller decides method


def run_benchmark():
    dataset = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    in_scope = [q for q in dataset if q["in_scope"]]
    out_of_scope = [q for q in dataset if not q["in_scope"]]

    print(f"\n{'='*70}")
    print(f"  Retrieval Benchmark — Northstar Operations Policy Corpus")
    print(f"{'='*70}")
    print(f"  {len(dataset)} questions  |  {len(in_scope)} in-scope  |  {len(out_of_scope)} OOD/adversarial")
    print(f"  Top-k = {TOP_K}  |  Semantic min_similarity = {SEM_MIN_SIMILARITY}")
    print()

    methods = {
        "BM25": lambda q: bm25_search(q, top_k=TOP_K),
        "Semantic": lambda q: semantic_search(q, top_k=TOP_K, min_similarity=SEM_MIN_SIMILARITY),
        "Hybrid": lambda q: hybrid_search(q, top_k=TOP_K, min_similarity=SEM_MIN_SIMILARITY),
        "Hybrid+Rerank": lambda q: reranked_search(q, top_k=TOP_K, min_similarity=SEM_MIN_SIMILARITY, candidate_k=20),
    }

    method_results = {}   # method -> list of result lists (in-scope only)
    method_latency = {}   # method -> avg ms per query (in-scope only)
    method_ood = {}       # method -> list of result lists (OOD only)

    for method_name, retriever in methods.items():
        # --- in-scope ---
        results_per_q = []
        t0 = time.perf_counter()
        for q in in_scope:
            results_per_q.append(retriever(q["question"]))
        elapsed_ms = (time.perf_counter() - t0) * 1000
        method_results[method_name] = results_per_q
        method_latency[method_name] = elapsed_ms / len(in_scope)

        # --- OOD/adversarial ---
        ood_results = []
        for q in out_of_scope:
            ood_results.append(retriever(q["question"]))
        method_ood[method_name] = ood_results

    # --- Compute metrics for in-scope ---
    print(f"  {'Method':<12} {'Hit@1':>7} {'Hit@3':>7} {'Hit@5':>7} {'MRR':>7} {'Avg ms':>8}")
    print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*8}")

    for method_name in methods:
        results_per_q = method_results[method_name]
        exp_docs    = [q["expected_document_id"]    for q in in_scope]
        exp_secs    = [q["expected_section_contains"] for q in in_scope]

        h1 = sum(_is_hit(res[:1], d, s) for res, d, s in zip(results_per_q, exp_docs, exp_secs))
        h3 = sum(_is_hit(res[:3], d, s) for res, d, s in zip(results_per_q, exp_docs, exp_secs))
        h5 = sum(_is_hit(res[:5], d, s) for res, d, s in zip(results_per_q, exp_docs, exp_secs))
        mrr = _mrr(results_per_q, exp_docs, exp_secs)
        n = len(in_scope)

        print(f"  {method_name:<12} {h1/n:>6.1%} {h3/n:>6.1%} {h5/n:>6.1%} {mrr:>6.3f} {method_latency[method_name]:>7.1f}ms")

    # --- OOD false-positive rate (BM25 always returns; semantic/hybrid filter by score)
    print()
    print(f"  OOD / Adversarial  ({len(out_of_scope)} questions — lower false-positive rate is better)")
    print(f"  {'Method':<12} {'FP (BM25: any result | Sem/Hybrid: score>{OOD_FP_THRESHOLD})':>52}")
    print(f"  {'-'*12} {'-'*52}")

    for method_name in methods:
        ood_res = method_ood[method_name]
        if method_name == "BM25":
            # BM25 always returns something; FP = returned at least 1 result with score > 0
            fps = sum(1 for res in ood_res if res and res[0].similarity_score > 0)
        else:
            fps = sum(1 for res in ood_res if res and res[0].similarity_score > OOD_FP_THRESHOLD)
        print(f"  {method_name:<12}  {fps}/{len(out_of_scope)} false positives  ({fps/len(out_of_scope):.0%})")

    # --- Per-category breakdown (Hybrid, Hit@3) ---
    print()
    print(f"  Hybrid Hit@3 by category")
    print(f"  {'-'*40}")

    hybrid_res = method_results["Hybrid"]
    categories = sorted({q["category"] for q in in_scope})
    for cat in categories:
        cat_qs  = [(i, q) for i, q in enumerate(in_scope) if q["category"] == cat]
        hits = sum(
            _is_hit(hybrid_res[i][:3], q["expected_document_id"], q["expected_section_contains"])
            for i, q in cat_qs
        )
        n_cat = len(cat_qs)
        bar = "█" * hits + "░" * (n_cat - hits)
        print(f"  {cat:<22} {hits}/{n_cat}  {bar}  ({hits/n_cat:.0%})")

    # --- Misses: which in-scope questions does Hybrid miss at k=3? ---
    print()
    print(f"  Hybrid misses at k=3 (in-scope questions not retrieved):")
    hybrid_res = method_results["Hybrid"]
    any_miss = False
    for i, q in enumerate(in_scope):
        if not _is_hit(hybrid_res[i][:3], q["expected_document_id"], q["expected_section_contains"]):
            any_miss = True
            top_doc = hybrid_res[i][0].document_id if hybrid_res[i] else "—"
            print(f"    [{q['category']}] {q['question'][:70]}")
            print(f"      expected: {q['expected_document_id']} / '{q['expected_section_contains']}'")
            print(f"      got:      {top_doc} (top result)")
    if not any_miss:
        print("    none — all in-scope questions hit at k=3")

    print()
    print(f"{'='*70}\n")


if __name__ == "__main__":
    run_benchmark()
