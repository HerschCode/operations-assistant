# Generation Fine-Tune: QLoRA Domain Adaptation

> **Status: negative result — train/test leakage detected.**
> The experiment documented here is preserved as a methodological record. The +17.9 pp
> improvement shown in the evaluation table was driven by memorization, not
> generalization. See [Leakage Analysis](#leakage-analysis) below and the corrected
> experiment in [tool-selection-finetune.md](tool-selection-finetune.md).

---

## Goal (original)

Fine-tune `Qwen2.5-0.5B-Instruct` on Northstar Manufacturing's P2P policy corpus so
the model can answer procurement, SLA, and escalation questions **directly from its
weights** without needing the live retrieval system.

---

## Leakage Analysis

The 130 in-scope policy questions map onto only **~58 unique policy chunks** (many
questions refer to the same section). A row-level 80/20 split assigns questions to
train/test independently, so the same chunk text ends up in both splits.

Result: **78.6% of test answers (22/28) are character-identical to a training
answer**. The +17.9 pp gain reflects the model recognising text it memorised during
training, not domain generalisation to new questions.

The leakage check is now enforced in CI:

```python
# scripts/prepare_generation_data.py
_check_leakage(train, test)   # raises ValueError if train and test share answers or citations
```

Running `python -m scripts.prepare_generation_data` will raise:

```
ValueError: Train/test leakage detected: 22 test answers and 28 test citations
appear in the train set. Use a chunk-deduped split (split by citation) instead.
```

**Corrected number**: A valid generation fine-tune would require a citation-deduped
split (split at the chunk level so every chunk appears in exactly one split). With
only 58 unique chunks, this yields a test set of ~12 chunks / ~26 questions — too
small for reliable estimates. The experiment was therefore re-framed to
**tool-selection fine-tuning** where a template-ID split is leak-free by construction.
See `docs/tool-selection-finetune.md`.

---

## Data

Source: `data/evaluation/eval_dataset.json` — 130 in-scope policy questions.

For each question, `search_policy_documents` (the live hybrid retrieval stack) returns
the highest-scoring chunk. The answer is formatted as:

```
According to {citation}: {chunk_text}
```

Row-level split 80/20 stratified by category (**leaky — see above**):

| Category             | Train | Test |
|----------------------|------:|-----:|
| lookup               |    24 |    6 |
| numerical            |    20 |    5 |
| policy_interpretation|    20 |    5 |
| multi_hop            |    17 |    5 |
| paraphrase           |    13 |    4 |
| ambiguous            |     8 |    3 |
| **Total**            |**102**|**28**|

---

## Training Setup

| Parameter           | Value                          |
|---------------------|-------------------------------|
| Base model          | Qwen/Qwen2.5-0.5B-Instruct    |
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

Trainable parameters: **4,399,104** (0.88% of 498M — LoRA adapter only, base model frozen)
Training time: **129s**
Final train loss: **1.78** (3.63 → 1.13 across 39 optimizer steps)

---

## Evaluation (invalid — leaked split)

Metric: **keyword recall** — fraction of reference keywords appearing in generated
response. A question is a "hit" if recall ≥ 0.50.

| Model       | Hit Rate | n  |
|-------------|:--------:|---:|
| Base        |  32.1%   | 28 |
| Fine-tuned  |  50.0%   | 28 |
| Δ           | **+17.9pp** ← **invalid; memorization artefact** | |

The improvement cannot be attributed to generalization because 22/28 test answers
were seen verbatim during training. The corrected figure from a leak-free eval is
reported in `docs/tool-selection-finetune.md`.

Full per-example detail (archived): `data/evaluation/generation_finetune_results.json`

---

## What Was Learned

1. **Row-level splits on a policy corpus are almost always leaky.** When many questions
   reference the same document section, the correct split unit is the *chunk* (citation),
   not the question row.

2. **Knowledge-in-weights is the wrong objective when RAG exists.** Fine-tuning a model
   to memorise retrieved text competes with the retrieval system. The correct
   fine-tuning objective when RAG is in the loop is **tool selection / routing** — teach
   the model *which tool to call and with what arguments*, not what the tool will return.

3. **QLoRA + Qwen2.5-0.5B runs in 129s on an RTX 4060.** The infrastructure is sound;
   only the task framing and split strategy needed correction.

---

## Reproduction

```bash
# Attempts to build the dataset — will raise ValueError due to leakage
python -m scripts.prepare_generation_data

# To reproduce the (invalid) fine-tuned eval result from the archived adapter:
python -m scripts.evaluate_generation
```

The corrected experiment (tool-selection fine-tuning with template-ID split) is
reproduced via:

```bash
python -m scripts.prepare_tool_selection_data
python -m scripts.finetune_tool_selection
python -m scripts.evaluate_tool_selection
```
