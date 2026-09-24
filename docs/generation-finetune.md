# Generation Fine-Tune: QLoRA Domain Adaptation

## Goal

Fine-tune `Qwen2.5-1.5B-Instruct` on Northstar Manufacturing's P2P policy corpus so
the model can answer procurement, SLA, and escalation questions **directly from its
weights** without needing the live retrieval system. This demonstrates:

1. **Domain adaptation via QLoRA** — 4-bit NF4 quantization + LoRA adapters make it
   feasible to fine-tune a 1.5B model on a single 8GB GPU.
2. **Data construction from a RAG corpus** — using the same retrieval stack as the live
   assistant to ground every training example in retrieved policy text (no hallucinated
   training labels, no separate LLM calls for data generation).
3. **Honest before/after evaluation** — keyword-recall metric on a held-out test split
   that the model never trained on.

---

## Data

Source: `data/evaluation/eval_dataset.json` — 130 in-scope policy questions.

For each question, `search_policy_documents` (the live hybrid retrieval stack) returns
the highest-scoring chunk. The answer is formatted as:

```
According to {citation}: {chunk_text}
```

Split 80/20 stratified by category (see `data/finetune/stats.json`):

| Category             | Train | Test |
|----------------------|------:|-----:|
| lookup               |    24 |    6 |
| numerical            |    20 |    5 |
| policy_interpretation|    20 |    5 |
| multi_hop            |    17 |    5 |
| paraphrase           |    13 |    4 |
| ambiguous            |     8 |    3 |
| **Total**            |**102**|**28**|

No LLM API calls used for data generation. The training answers are directly grounded
in retrieved policy chunks, so the fine-tuned model learns to produce the same
content the RAG assistant would retrieve — without the retrieval step.

---

## Training Setup

| Parameter           | Value                          |
|---------------------|-------------------------------|
| Base model          | Qwen/Qwen2.5-1.5B-Instruct    |
| Quantisation        | 4-bit NF4 (bitsandbytes)      |
| Compute dtype       | bfloat16                      |
| LoRA rank r         | 8                             |
| LoRA alpha          | 16  (α/r = 2)                 |
| LoRA target modules | q/k/v/o_proj, gate/up/down_proj |
| LoRA dropout        | 0.05                          |
| Epochs              | 3                             |
| Batch size          | 4 (×2 grad accum = 8 eff.)    |
| Learning rate       | 2e-4                          |
| LR scheduler        | cosine                        |
| Max seq length      | 512                           |
| Optimiser           | paged_adamw_8bit              |
| Hardware            | NVIDIA RTX 4060 Laptop 8 GB   |

Trainable parameters: **TBD** (LoRA adapter only — base model frozen)
Training time: **TBD**s
Final train loss: **TBD**

Full training metrics: `models/generation_adapter/training_metrics.json`
LoRA adapter weights: `models/generation_adapter/adapter/`

---

## Evaluation

Metric: **keyword recall** — for each test question, extract numbers, capitalized
policy terms, and domain keywords from the reference answer; check what fraction
appear in the model's generated response. A question is a "hit" if recall ≥ 0.50.

The base model (no fine-tuning) cannot answer Northstar's fictional policies from
pre-training knowledge, so this is a clean domain-adaptation test: any improvement
is directly attributable to the QLoRA fine-tuning.

| Model       | Hit Rate | n |
|-------------|:--------:|--:|
| Base        |   **TBD** | 28 |
| Fine-tuned  |   **TBD** | 28 |
| Δ           |   **TBD** |    |

Per-category breakdown: see `data/evaluation/generation_finetune_results.json`.

---

## Limitations and Caveats

- **Training set size**: 102 examples is small for instruction tuning. Multi-hop and
  ambiguous categories are expected to generalise less than lookup/paraphrase.
- **Metric conservatism**: keyword recall measures token presence, not semantic
  correctness. A model that paraphrases correctly but uses different wording scores
  lower than one that quotes verbatim.
- **Answer quality**: training labels are formatted as "According to X: {raw chunk
  text}" — the fine-tuned model learns to quote policy sections, not to synthesise
  them into flowing prose. Adequate for domain retrieval; not natural conversation.
- **Not a replacement for RAG**: the fine-tuned model is evaluated on questions whose
  answers appear in the training distribution. Novel policy updates would require
  re-training or a retrieval step.

---

## Reproduction

```bash
# 1. Build training data (CPU only, no API keys)
python -m scripts.prepare_generation_data

# 2. Fine-tune (requires CUDA, ~8 GB VRAM)
python -m scripts.finetune_generation

# 3. Evaluate base vs. fine-tuned
python -m scripts.evaluate_generation
```
