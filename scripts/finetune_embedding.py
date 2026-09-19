"""
Offline experiment: does fine-tuning the retrieval embedding model on this corpus help?

SCOPE, stated plainly: this fine-tunes the *embedding* model (all-MiniLM-L6-v2) with
MultipleNegativesRankingLoss on (question, correct-chunk) pairs. It does NOT fine-tune a
generation LLM (no GPU / no fine-tuning API budget here) and the result is NOT deployed:
the served retriever still uses the stock ONNX model. This is a measured experiment.

Leakage control -- the whole point of doing this carefully with ~130 labelled questions:
  * question split: random 70/30 over questions (test questions never seen in training,
    but their *documents* were). Optimistic setting.
  * document split: every question about a held-out set of documents is test-only, so the
    model is evaluated on documents whose questions it never trained on. Pessimistic /
    generalisation setting.
Candidates at test time are ALL chunks in the corpus, in both settings (no shrinking the
haystack). Metrics are Hit@1 / Hit@3 / MRR with the same section-level matching as
scripts/benchmark_retrieval.py, dense retrieval only (cosine), and a paired bootstrap CI on
the per-question difference (fine-tuned minus base).

Run:  python -X utf8 -m scripts.finetune_embedding [--seeds 5] [--epochs 4]
Writes docs/embedding-finetune-raw-result.json; the model itself is not saved.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

from src.ingestion.chunker import chunk_text
from src.ingestion.document_loader import load_all_documents

ROOT = Path(__file__).parent.parent
EVAL_PATH = ROOT / "data/evaluation/eval_dataset.json"
OUT_PATH = ROOT / "docs/embedding-finetune-raw-result.json"
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_corpus():
    chunks = []  # (doc_id, section_title, text)
    for doc in load_all_documents(ROOT / "data/documents"):
        for c in chunk_text(doc.text):
            chunks.append((doc.document_id, c.section_title or "", c.text))
    return chunks


def positives(question, chunks):
    sec = (question["expected_section_contains"] or "").lower()
    return [i for i, (d, s, _) in enumerate(chunks)
            if d == question["expected_document_id"] and sec in s.lower()]


def evaluate(model, questions, chunks):
    """Per-question (hit1, hit3, rr) arrays. Dense cosine retrieval over all chunks."""
    c_emb = model.encode([c[2] for c in chunks], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    q_emb = model.encode([q["question"] for q in questions], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    order = np.argsort(-(q_emb @ c_emb.T), axis=1)
    h1, h3, rr = [], [], []
    for q, ranked in zip(questions, order):
        pos = set(positives(q, chunks))
        rank = next((r for r, idx in enumerate(ranked[:5], 1) if idx in pos), None)
        h1.append(float(rank == 1)); h3.append(float(rank is not None and rank <= 3)); rr.append(1.0 / rank if rank else 0.0)
    return np.array(h1), np.array(h3), np.array(rr)


def train(train_qs, chunks, epochs, batch_size, lr, seed):
    import torch
    from sentence_transformers import SentenceTransformer, losses
    torch.manual_seed(seed)
    model = SentenceTransformer(BASE_MODEL, device="cpu")
    loss_fn = losses.MultipleNegativesRankingLoss(model)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    rng = random.Random(seed)
    pairs = [(q["question"], p) for q in train_qs for p in positives(q, chunks)[:1]]
    model.train()
    for _ in range(epochs):
        rng.shuffle(pairs)
        # in-batch negatives are only valid if no two pairs in a batch share a positive chunk
        batches, cur, used = [], [], set()
        for pr in pairs:
            if pr[1] in used or len(cur) == batch_size:
                batches.append(cur); cur, used = [], set()
            cur.append(pr); used.add(pr[1])
        if cur:
            batches.append(cur)
        for batch in batches:
            if len(batch) < 2:
                continue
            feats = [model.tokenize([b[0] for b in batch]), model.tokenize([chunks[b[1]][2] for b in batch])]
            feats = [{k: v for k, v in f.items()} for f in feats]
            loss = loss_fn(feats, None)
            loss.backward(); opt.step(); opt.zero_grad()
    model.eval()
    return model


def paired_bootstrap(diff, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    means = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise(name, base, ft):
    out = {}
    for label, b, f in zip(("hit1", "hit3", "mrr"), base, ft):
        diff = f - b
        lo, hi = paired_bootstrap(diff)
        out[label] = {"base": float(b.mean()), "finetuned": float(f.mean()), "delta": float(diff.mean()), "ci95": [lo, hi]}
    print(f"  {name:<10} n={len(base[0]):>3}  " + "  ".join(
        f"{k}: {v['base']:.3f}->{v['finetuned']:.3f} ({v['delta']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}])" for k, v in out.items()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    from sentence_transformers import SentenceTransformer
    chunks = build_corpus()
    data = [q for q in json.loads(EVAL_PATH.read_text(encoding="utf-8")) if q["in_scope"] and positives(q, [c for c in chunks])]
    doc_ids = sorted({q["expected_document_id"] for q in data})
    print(f"{len(chunks)} chunks | {len(data)} usable in-scope questions | {len(doc_ids)} documents")
    base_model = SentenceTransformer(BASE_MODEL, device="cpu")

    results = {"config": vars(args), "n_chunks": len(chunks), "n_questions": len(data), "runs": []}
    pooled = {"question": [[], [], []], "document": [[], [], []]}
    pooled_base = {"question": [[], [], []], "document": [[], [], []]}

    for seed in range(args.seeds):
        rng = random.Random(seed)
        # question-level split
        qs = data[:]; rng.shuffle(qs)
        cut = int(0.7 * len(qs))
        splits = {"question": (qs[:cut], qs[cut:])}
        # document-level split: hold out ~30% of documents
        held = set(rng.sample(doc_ids, max(2, round(0.3 * len(doc_ids)))))
        splits["document"] = ([q for q in data if q["expected_document_id"] not in held],
                              [q for q in data if q["expected_document_id"] in held])
        for split_name, (tr, te) in splits.items():
            base = evaluate(base_model, te, chunks)
            ft_model = train(tr, chunks, args.epochs, args.batch_size, args.lr, seed)
            ft = evaluate(ft_model, te, chunks)
            print(f"seed {seed}")
            results["runs"].append({"seed": seed, "split": split_name, "n_train": len(tr), "n_test": len(te),
                                    **summarise(split_name, base, ft)})
            for i in range(3):
                pooled[split_name][i].append(ft[i]); pooled_base[split_name][i].append(base[i])

    print("\nPooled across seeds (test questions repeat across seeds, so CIs are optimistic-narrow):")
    results["pooled"] = {}
    for split_name in pooled:
        b = tuple(np.concatenate(x) for x in pooled_base[split_name])
        f = tuple(np.concatenate(x) for x in pooled[split_name])
        results["pooled"][split_name] = summarise(split_name, b, f)

    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
