"""
QLoRA fine-tune Qwen2.5-1.5B-Instruct on the P2P operations policy dataset.

Uses 4-bit NF4 quantization (bitsandbytes) + LoRA adapters (peft) targeting all
attention and feed-forward projection layers. Training uses transformers.Trainer
with a causal LM objective over the full formatted sequence.

Hardware: designed for 8 GB VRAM (RTX 4060 / T4). With the default settings the
peak VRAM usage is ~4-5 GB, leaving headroom for activation memory.

Run:
  python -m scripts.finetune_generation [options]

Options:
  --model          HF model id (default: Qwen/Qwen2.5-1.5B-Instruct)
  --train-data     JSONL path (default: data/finetune/train.jsonl)
  --output-dir     Adapter + metrics destination (default: models/generation_adapter)
  --epochs         Training epochs (default: 3)
  --batch-size     Per-device batch (default: 4, effective=8 with grad accum 2)
  --lr             Learning rate (default: 2e-4)
  --r              LoRA rank (default: 8)
  --lora-alpha     LoRA alpha (default: 16)
  --max-length     Max token length per example (default: 512)
  --no-cuda        Force CPU (for debugging only -- QLoRA requires CUDA in practice)
"""
import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_TRAIN = ROOT / "data/finetune/train.jsonl"
DEFAULT_OUT = ROOT / "models/generation_adapter"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--train-data", default=str(DEFAULT_TRAIN))
    ap.add_argument("--output-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--no-cuda", action="store_true")
    return ap.parse_args()


def load_dataset(jsonl_path: str, tokenizer, max_length: int):
    """Load JSONL, apply chat template, tokenize. Returns HuggingFace Dataset."""
    import json as _json
    from datasets import Dataset

    examples = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = _json.loads(line)
            text = tokenizer.apply_chat_template(
                item["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            examples.append({"text": text, "meta": item.get("meta", {})})

    dataset = Dataset.from_list(examples)

    def _tokenize(batch):
        tokenized = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        # Full-sequence loss: train on system+user+assistant together.
        # For response-only loss masking, use trl's DataCollatorForCompletionOnlyLM.
        tokenized["labels"] = [list(ids) for ids in tokenized["input_ids"]]
        return tokenized

    return dataset.map(_tokenize, batched=True, remove_columns=["text", "meta"])


def main():
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device_map = "auto" if use_cuda else "cpu"
    print(f"Device: {'CUDA (' + torch.cuda.get_device_name(0) + ')' if use_cuda else 'CPU'}")
    print(f"Model:  {args.model}")
    print(f"Train:  {args.train_data}")
    print(f"Out:    {args.output_dir}")
    print()

    # ── tokenizer ──────────────────────────────────────────────────────────────
    print("Loading tokenizer ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── model (4-bit QLoRA) ────────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ) if use_cuda else None

    print("Loading model ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        dtype=torch.bfloat16 if use_cuda else torch.float32,
    )

    if use_cuda:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # ── LoRA adapters ──────────────────────────────────────────────────────────
    # Target all linear projection layers in attention (q/k/v/o) and MLP (gate/up/down).
    # r=8 with alpha=16 gives alpha/r=2, a standard scaling factor for instruction tuning.
    lora_cfg = LoraConfig(
        r=args.r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    print()

    # ── dataset ────────────────────────────────────────────────────────────────
    print("Tokenising training data ...", flush=True)
    train_dataset = load_dataset(args.train_data, tokenizer, args.max_length)
    print(f"Training examples: {len(train_dataset)}")
    print()

    # ── training ───────────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,           # effective batch = batch_size × 2
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=use_cuda and not torch.cuda.is_bf16_supported(),
        bf16=use_cuda and torch.cuda.is_bf16_supported(),
        logging_steps=5,
        save_strategy="no",                      # save adapter manually after training
        optim="paged_adamw_8bit" if use_cuda else "adamw_torch",
        report_to="none",
        dataloader_pin_memory=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print("Training ...", flush=True)
    t0 = time.monotonic()
    train_result = trainer.train()
    elapsed = round(time.monotonic() - t0, 1)
    print(f"Training complete in {elapsed}s", flush=True)

    # ── save ───────────────────────────────────────────────────────────────────
    adapter_path = out_dir / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"Adapter saved to {adapter_path.relative_to(ROOT)}")

    metrics = {
        "model": args.model,
        "lora_r": args.r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_length": args.max_length,
        "n_train": len(train_dataset),
        "train_loss": round(train_result.training_loss, 4),
        "elapsed_seconds": elapsed,
        "device": torch.cuda.get_device_name(0) if use_cuda else "cpu",
        "trainable_params": model.num_parameters(only_trainable=True),
        "total_params": model.num_parameters(),
    }
    metrics_path = out_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Metrics:  {metrics_path.relative_to(ROOT)}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
