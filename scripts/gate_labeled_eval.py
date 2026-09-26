"""Evaluate the answer gate as a CLASSIFIER against labeled answers, not as a "coverage" number.

Why: the earlier calibration counted every blocked in-domain answer as either a caught hallucination or an
over-block using the NLI scores themselves (circular), and its OOD set had empty answers (the gate fired on the
empty-answer rule, never on NLI). Here every answer has a label that does not come from the gate:

- correct (32): the stored in-domain answers (data/evaluation/faithfulness_results.json), each hand-checked
  against the policy documents (data/evaluation/gate_labels.json; same author -- disclosed).
- wrong_fact (32): each correct answer with ONE claim deterministically mutated (a number changed, or a key term
  swapped: team, frequency, yes/no, role) -- the realistic in-domain hallucination.
- off_context (32): each correct answer scored against the chunks retrieved for a DIFFERENT, unrelated question --
  an answer the retrieved evidence does not support.

Gates compared (an answer PASSES or is BLOCKED):
- nli@t: deployed gate. Sentence grounded if max NLI entailment >= 0.5; answer passes if the fraction of grounded
  sentences >= t (deployed t = 0.05).
- support@r: new lexical claim-support gate (src/evaluation/claim_support.py). Every number must occur in the chunks attached to the same
  unit, every key term (name/team/role/frequency) must occur, and >= r of its content words must too; answer passes if every sentence is supported.
- support+nli: support@r AND no sentence with NLI contradiction >= 0.9 unless it is also lexically supported
  (i.e. NLI can veto only unsupported sentences).
r is chosen on the ODD question ids only; all results are reported on the EVEN ids (held out) and on all.

Run: python -m scripts.gate_labeled_eval  -> reports/gate_labeled_eval.json
"""
import json
import re
from pathlib import Path

import numpy as np

from src.evaluation.claim_support import sentence_support, split_sentences

MUTATIONS = [  # (pattern, replacement) tried in order; first that changes the text is used
    (r"\bVendor Risk team\b", "Finance team"),
    (r"\bquarterly\b", "monthly"),
    (r"\bProcurement leadership\b", "the CFO"),
    (r"\bOperations leadership\b", "the Legal department"),
    (r"\bOperations Director\b", "Chief Executive"),
    (r"\$10,000", "$25,000"),
    (r"\$500", "$2,000"),
    (r"\bthree\b", "five"),
    (r"\b14\b", "21"), (r"\b10\b", "15"), (r"\b5\b", "7"), (r"\b3\b", "6"), (r"\b2\b", "4"),
    (r"^Yes\.", "No."), (r"^No\.", "Yes."),
    (r"\bincluded\b", "excluded"), (r"\bdoes not count\b", "counts"),
    (r"\boriginal approval chain\b", "Legal department sign-off"), (r"\bkeep\b|\bretain\b", "retains"),
    (r"\bweekly\b", "annual"), (r"\bon \*?\*?hold\b", "cancelled"), (r"\bsupplier\b", "auditor"), (r"\bbudget owner\b", "external auditor"),
]


def mutate(answer: str) -> str:
    for pat, rep in MUTATIONS:
        new = re.sub(pat, rep, answer, count=1)
        if new != answer:
            return new
    raise ValueError(f"no mutation applies: {answer[:80]}")


def nli_scores(sentences, chunks):
    from src.evaluation.faithfulness import _load_nli_model

    if not sentences:
        return np.zeros((0, 3))
    pairs = [(s, c) for s in sentences for c in chunks]
    sc = np.array(_load_nli_model().predict(pairs, apply_softmax=True)).reshape(len(sentences), len(chunks), 3)
    return np.stack([sc[:, :, 0].max(1), sc[:, :, 1].max(1)], axis=1)   # [max contradiction, max entailment]


def features(answer, chunks):
    sents = split_sentences(answer)
    nli = nli_scores(sents, chunks)
    sup = [sentence_support(s, chunks) for s in sents]
    return {"n": len(sents), "entail": nli[:, 1].tolist() if len(sents) else [], "contra": nli[:, 0].tolist() if len(sents) else [],
            "num_ok": [x["numbers_supported"] and x["key_terms_supported"] for x in sup],
            "recall": [x["content_recall"] for x in sup]}


def gate_nli(f, t):
    if f["n"] == 0:
        return False
    return np.mean([e >= 0.5 for e in f["entail"]]) >= t


def gate_support(f, r):
    return f["n"] > 0 and all(n and c >= r for n, c in zip(f["num_ok"], f["recall"]))


def gate_combo(f, r):
    if not gate_support(f, r):
        return False
    return True  # contradiction veto only applies to unsupported sentences, which gate_support already blocks


def metrics(rows, gate):
    out = {}
    for lab in ("correct", "wrong_fact", "off_context"):
        sel = [gate(x["f"]) for x in rows if x["label"] == lab]
        out[f"{lab}_pass_rate"] = round(float(np.mean(sel)), 3) if sel else None
    out["n"] = len(rows)
    return out


def main():
    from src.retrieval.search import reranked_search

    res = json.load(open("data/evaluation/faithfulness_results.json", encoding="utf-8"))["results"]
    labels = {x["id"]: x["label"] for x in json.load(open("data/evaluation/gate_labels.json", encoding="utf-8"))["labels"]}
    assert all(labels[r["id"]] == "correct" for r in res)
    chunks = {r["id"]: [c.text for c in reranked_search(r["question"], top_k=5)] for r in res}
    ids = [r["id"] for r in res]
    # off-context partner: the question whose retrieved chunks share the fewest documents' text with this one
    def overlap(a, b):
        return len(set(chunks[a]) & set(chunks[b]))
    partner = {i: min((j for j in ids if j != i), key=lambda j: (overlap(i, j), (j - i) % len(ids))) for i in ids}

    rows = []
    for r in res:
        a = r["answer"]
        rows.append({"id": r["id"], "label": "correct", "answer": a, "f": features(a, chunks[r["id"]])})
        m = mutate(a)
        rows.append({"id": r["id"], "label": "wrong_fact", "answer": m, "f": features(m, chunks[r["id"]])})
        rows.append({"id": r["id"], "label": "off_context", "answer": a, "partner": partner[r["id"]],
                     "f": features(a, chunks[partner[r["id"]]])})
        print(r["id"], flush=True)

    tune = [x for x in rows if x["id"] % 2 == 1]
    test = [x for x in rows if x["id"] % 2 == 0]
    grid = [round(v, 2) for v in np.arange(0.0, 0.96, 0.05)]

    def score(m):  # balanced accuracy: pass correct, block the two wrong kinds
        return (m["correct_pass_rate"] + (1 - m["wrong_fact_pass_rate"]) + (1 - m["off_context_pass_rate"])) / 3

    tuned = {r: score(metrics(tune, lambda f, r=r: gate_support(f, r))) for r in grid}
    best_r = max(tuned, key=lambda r: (tuned[r], -r))
    report = {"design": __doc__.split("Run:")[0], "chosen_support_recall": best_r,
              "tuning_curve_odd_ids": tuned, "gates": {}}
    for name, g in [("nli@0.05 (deployed)", lambda f: gate_nli(f, 0.05)), ("nli@0.5", lambda f: gate_nli(f, 0.5)),
                    ("nli@1.0", lambda f: gate_nli(f, 1.0)), (f"support@{best_r}", lambda f: gate_support(f, best_r)),
                    ("no gate", lambda f: f["n"] > 0)]:
        report["gates"][name] = {"held_out_even_ids": metrics(test, g), "all": metrics(rows, g)}
    report["rows"] = [{k: v for k, v in x.items()} for x in rows]
    Path("reports").mkdir(exist_ok=True)
    Path("reports/gate_labeled_eval.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"chosen r = {best_r}")
    for name, v in report["gates"].items():
        print(f"{name:22s} held-out {v['held_out_even_ids']}   all {v['all']}")


if __name__ == "__main__":
    main()
