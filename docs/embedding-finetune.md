# Embedding fine-tuning experiment (offline, not deployed)

**Scope.** This fine-tunes the *retrieval embedding model* (`all-MiniLM-L6-v2`) with
MultipleNegativesRankingLoss on (question, correct-chunk) pairs from
`data/evaluation/eval_dataset.json`. It is **not** fine-tuning of a generation LLM (that is a separate experiment) and
nothing from it is deployed; the served retriever still uses the
stock ONNX model. Script: `scripts/finetune_embedding.py`; raw output:
`docs/embedding-finetune-raw-result.json`.

**Setup.** 130 in-scope questions, 58 chunks (all chunks are candidates at test time), dense
cosine retrieval, section-level matching as in `benchmark_retrieval.py`. 4 epochs, batch 16,
lr 2e-5, 5 random seeds, two split types:

* *question split* (70/30 over questions; test documents were seen in training)
* *document split* (~30% of documents held out; every question about them is test-only)

**Result (pooled over 5 seeds; paired bootstrap 95% CI on fine-tuned minus base).**

| Split | Metric | Base | Fine-tuned | Delta [95% CI] |
|---|---|---|---|---|
| Question | Hit@1 | 0.703 | 0.708 | +0.005 [-0.051, +0.062] |
| Question | Hit@3 | 0.856 | 0.887 | +0.031 [-0.010, +0.072] |
| Question | MRR | 0.783 | 0.801 | +0.018 [-0.021, +0.058] |
| Document | Hit@1 | 0.726 | 0.749 | +0.023 [-0.019, +0.065] |
| Document | Hit@3 | 0.870 | 0.874 | +0.005 [-0.023, +0.033] |
| Document | MRR | 0.799 | 0.813 | +0.014 [-0.009, +0.038] |

**Conclusion: no reliable improvement.** Every pooled CI includes zero. Individual seeds swing
both ways (for example one question-split seed lost 12.8pp Hit@1; one document-split seed
gained 10.5pp), which is what a ~90-example training set produces. The pooled CIs are also
optimistic-narrow because the same questions recur across seeds. With this little labelled
data the fine-tune is not worth deploying, and the stock model stays. What would change the
answer is more (and independently written) training questions, not more epochs; that is a
data problem, not a tuning problem. Also note the base model here is dense-only, which is why
its Hit@1 differs slightly from the full-benchmark Semantic row (different candidate scoring
path: raw cosine over all chunks, no similarity floor).
