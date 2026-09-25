"""
QLoRA fine-tune Qwen2.5-0.5B-Instruct on tool-selection data.

Task: given a user question, output a JSON tool-call (or array of tool-calls).
Training data: data/finetune/tool_train.jsonl (split by template_id — no leakage).

Run:
  python -m scripts.finetune_tool_selection [--model ...] [--output-dir ...]

Produces:
  models/tool_selection_adapter/adapter/   -- LoRA weights
  models/tool_selection_adapter/training_metrics.json
"""
import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_TRAIN = ROOT / "data/finetune/tool_train.jsonl"
DEFAULT_OUT = ROOT / "models/tool_selection_adapter"


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
                item["messages"], tokenize=False, add_generation_prompt=False,
            )
            examples.append({"text": text})

    dataset = Dataset.from_list(examples)

    def _tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length, padding=False)

    return dataset.map(_tokenize, batched=True, remove_columns=["text"])


def main():
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
        DataCollatorForLanguageModeling, TrainingArguments, Trainer,
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

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = (
        BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        if use_cuda else None
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb_config, device_map=device_map,
        trust_remote_code=True, dtype=torch.bfloat16 if use_cuda else torch.float32,
    )
    if use_cuda:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_cfg = LoraConfig(
        r=args.r, lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print("Tokenising training data ...", flush=True)
    train_dataset = load_dataset(args.train_data, tokenizer, args.max_length)
    print(f"Training examples: {len(train_dataset)}")

    training_args = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        weight_decay=0.01,
        fp16=use_cuda and not torch.cuda.is_bf16_supported(),
        bf16=use_cuda and torch.cuda.is_bf16_supported(),
        logging_steps=5,
        save_strategy="no",
        optim="paged_adamw_8bit" if use_cuda else "adamw_torch",
        report_to="none",
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8),
    )

    print("Training ...", flush=True)
    t0 = time.monotonic()
    train_result = trainer.train()
    elapsed = round(time.monotonic() - t0, 1)
    print(f"Training complete in {elapsed}s")

    adapter_path = out_dir / "adapter"
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))
    print(f"Adapter saved to {adapter_path.relative_to(ROOT)}")

    metrics = {
        "model": args.model,
        "task": "tool_selection",
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
    (out_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
