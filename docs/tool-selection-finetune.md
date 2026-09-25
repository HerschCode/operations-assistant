# Tool-Selection Fine-Tune: QLoRA on Structured Output

This is the corrected Phase 7 experiment. It supersedes the generation fine-tune
(see [generation-finetune.md](generation-finetune.md)) which had train/test leakage.

## Why tool selection, not generation

When a retrieval system already exists, fine-tuning a model to re-produce retrieved text
competes with RAG rather than complementing it. The correct objective is to teach the
model **which tool to call and with what arguments** — i.e., the routing decision the
agent must make before any retrieval happens.

The generation fine-tune also had 78.6% test-answer overlap with the training set
(same policy chunks appear in both splits), making its +17.9pp gain uninterpretable.
A tool-selection dataset built from templates split at the template level has zero leakage
by construction.

## Data

**Script:** `scripts/prepare_tool_selection_data.py`  
**Format:** ShareGPT JSONL — `(system_prompt + user_question) → JSON tool-call`

| File | Examples |
|------|----------|
| `data/finetune/tool_train.jsonl` | 388 |
| `data/finetune/tool_test.jsonl` | 112 |
| **Total** | **500** |

Source breakdown:
- **431 template examples**: 128 templates × paraphrases for all 9 registered tools
  (single-tool, 2-tool, 3-tool variants; argument value coverage for `segment`,
  `top_n`, `case_id`, `query`)
- **69 agent trace examples**: real questions from `agent_questions_v2.json` that have
  `expected_tools` defined, added verbatim

**Split strategy:** templates are assigned to train or test as whole groups
(`_split_by_template`), so no paraphrase variant of the same intent appears in both
splits. The leakage check (`_check_leakage`) runs after every split and is enforced in CI.

```
Template coverage: 102 train templates / 26 test templates
```

## Training setup

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen2.5-0.5B-Instruct |
| Quantisation | 4-bit NF4 (bitsandbytes) |
| Compute dtype | bfloat16 |
| LoRA rank r | 8 |
| LoRA alpha | 16 (α/r = 2) |
| LoRA target modules | q/k/v/o_proj, gate/up/down_proj |
| LoRA dropout | 0.05 |
| Epochs | 3 |
| Batch size | 4 (×2 grad accum = 8 eff.) |
| Learning rate | 2e-4 |
| LR scheduler | cosine |
| Max seq length | 512 |
| Optimiser | paged_adamw_8bit |
| Hardware | NVIDIA GeForce RTX 4060 Laptop GPU |

Trainable parameters: **4,399,104** (0.88% of 498M)  
Training time: **538s**  
Final train loss: **0.355** (3.61 → 0.082 across 147 steps)

Adapter: `models/tool_selection_adapter/adapter/` (tracked on HF Hub, not in git)

## Evaluation

**Script:** `scripts/evaluate_tool_selection.py`  
**Test set:** 181 examples (112 held-out templates + 69 agent traces)  
**Metrics:** json_valid, exact_tool_match, arg_match — all with bootstrap 95% CIs (10,000 resamples)

### Results

| Metric | Base | CI 95% | Fine-tuned | CI 95% | Δ |
|---|:---:|:---:|:---:|:---:|:---:|
| JSON valid | 85.1% | [79.6, 90.1] | **98.9%** | [97.2, 100.0] | +13.8 pp |
| Exact tool match | 33.7% | [27.1, 40.9] | **77.3%** | [71.3, 83.4] | +43.6 pp |
| Argument match | 26.5% | [20.4, 33.2] | **61.9%** | [54.7, 69.1] | +35.4 pp |
| p50 latency | 750 ms | — | 1,093 ms | — | +343 ms |

### Per-source breakdown

| Source | n | Base tool match | Fine-tuned tool match |
|---|:---:|:---:|:---:|
| Template test set | 99 | 35.4% | **90.9%** |
| Agent traces (real questions) | 82 | 31.7% | **61.0%** |

The fine-tuned model generalises well to held-out templates (90.9%) but shows a
larger gap on real agent traces (61.0%). The agent traces include multi-tool questions
and ambiguous phrasings that differ from template style; closing this gap would require
more diverse training examples.

### Why the base model scores so low

`Qwen2.5-0.5B-Instruct` is not fine-tuned for structured JSON output. It often
generates prose answers ("I'll help you check the cycle time...") rather than a JSON
tool call. After fine-tuning, the model learns the output format almost perfectly
(98.9% JSON validity) and the correct tool routing.

### Argument match gap

Exact tool match (77.3%) is meaningfully higher than arg match (61.9%), a 15.4 pp gap.
The model often selects the right tool but omits or mis-formats optional arguments
(e.g., `{"segment": "category"}` vs `{}`). This reflects the inherent ambiguity in
training examples where the same question appears with and without an argument depending
on the template variant.

### Latency

The fine-tuned model is 343ms slower at p50 (1093ms vs 750ms) because it generates
longer, well-formed JSON rather than short prose fragments. This is acceptable for a
routing call that precedes a tool invocation.

## Leakage check in CI

```bash
python -m pytest tests/test_phase7.py -v -k "leakage"
```

Catches both the old generation split (intentionally leaky, confirms the negative result)
and the new tool-selection split (verified clean).

## Reproduction

```bash
# 1. Build dataset (CPU, no API keys)
python -m scripts.prepare_tool_selection_data

# 2. Fine-tune (CUDA, ~538s on RTX 4060)
HF_HUB_DISABLE_XET=1 python -m scripts.finetune_tool_selection

# 3. Evaluate
HF_HUB_DISABLE_XET=1 python -m scripts.evaluate_tool_selection
```
