"""
Compare base Qwen2.5-1.5B-Instruct against the QLoRA fine-tuned adapter on the
held-out test split from data/finetune/test.jsonl.

Metric: keyword recall -- for each test example, extract the key tokens from the
reference answer (numbers, capitalised multi-word phrases, specific policy terms)
and check how many appear in the model's generated answer. A question is "hit" if
the recall is >= 0.5 (at least half the key tokens appear).

This is a domain-adaptation test: Northstar Manufacturing's specific policies are
fictional, so the base model genuinely cannot answer them from pre-training knowledge.
Any improvement from the fine-tuned model is attributable to the QLoRA training.

Output: data/evaluation/generation_finetune_results.json

Run: python -m scripts.evaluate_generation [--adapter models/generation_adapter/adapter]
"""
import argparse
import json
import re
import time
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
DEFAULT_ADAPTER = ROOT / "models/generation_adapter/adapter"
DEFAULT_TEST = ROOT / "data/finetune/test.jsonl"
DEFAULT_OUT = ROOT / "data/evaluation/generation_finetune_results.json"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    ap.add_argument("--test-data", default=str(DEFAULT_TEST))
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--max-length", type=int, default=512)
    return ap.parse_args()


# Stopwords to exclude from keyword extraction
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "do", "does", "for", "from", "had", "has", "have", "he", "her", "here",
    "how", "i", "if", "in", "is", "it", "its", "not", "of", "on", "or",
    "our", "per", "she", "so", "than", "that", "the", "their", "they",
    "this", "to", "was", "we", "were", "what", "when", "which", "who",
    "will", "with", "you",
}


def extract_keywords(text: str) -> set[str]:
    """Extract numbers and capitalised / domain-specific tokens from text."""
    tokens = set()
    # All numbers (including decimals and ranges like "5-10")
    for m in re.finditer(r"\b\d+(?:[.,]\d+)?(?:\s*[-–]\s*\d+)?\b", text):
        tokens.add(m.group().strip())
    # Capitalised words / sequences (policy titles, section headers)
    for m in re.finditer(r"\b[A-Z][a-zA-Z\-]{2,}\b", text):
        w = m.group()
        if w.lower() not in _STOPWORDS:
            tokens.add(w)
    # Lower-case domain terms that appear in policy language
    for m in re.finditer(r"\b(?:business\s+days?|approval|sla|breach|escalat\w+|supplier|"
                         r"procurement|consignment|match|renewal|exception|waiver|threshold)\b",
                         text, re.IGNORECASE):
        tokens.add(m.group().lower())
    return tokens


def keyword_recall(reference: str, generated: str) -> float:
    """Fraction of reference keywords found in generated text. 0.0–1.0."""
    ref_kw = extract_keywords(reference)
    if not ref_kw:
        return 1.0  # no keywords to match → trivially correct
    gen_lower = generated.lower()
    hits = sum(1 for kw in ref_kw if kw.lower() in gen_lower)
    return hits / len(ref_kw)


def load_test(path: str) -> list[dict]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def _load_base_model(model_id: str, device_map: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map=device_map,
        trust_remote_code=True, dtype=torch.bfloat16,
    )
    return model, tokenizer


def _load_finetuned_model(model_id: str, adapter_path: str, device_map: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map=device_map,
        trust_remote_code=True, dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, adapter_path)
    return model, tokenizer


def generate_answer(model, tokenizer, messages: list[dict], max_new_tokens: int, max_length: int) -> str:
    """Generate the assistant's response for the given messages (system + user)."""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_length,
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    # Decode only the new tokens (not the prompt)
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


HIT_THRESHOLD = 0.5


def evaluate_model(model, tokenizer, examples: list[dict], label: str, args) -> list[dict]:
    results = []
    for i, ex in enumerate(examples, 1):
        messages = ex["messages"]
        reference = next(m["content"] for m in messages if m["role"] == "assistant")
        # Strip assistant turn so the model has to generate it
        prompt_messages = [m for m in messages if m["role"] != "assistant"]

        t0 = time.monotonic()
        generated = generate_answer(model, tokenizer, prompt_messages, args.max_new_tokens, args.max_length)
        elapsed_ms = round((time.monotonic() - t0) * 1000, 1)

        recall = keyword_recall(reference, generated)
        hit = recall >= HIT_THRESHOLD
        results.append({
            "id": ex.get("meta", {}).get("id"),
            "category": ex.get("meta", {}).get("category"),
            "question": next(m["content"] for m in messages if m["role"] == "user"),
            "reference": reference,
            f"{label}_answer": generated,
            f"{label}_recall": round(recall, 3),
            f"{label}_hit": hit,
            "elapsed_ms": elapsed_ms,
        })
        status = "HIT " if hit else "MISS"
        print(f"  [{i:>3}/{len(examples)}] {status}  recall={recall:.2f}  {ex.get('meta', {}).get('category', '?')}")
    return results


def summarise(results: list[dict], label: str) -> dict:
    by_cat: dict[str, list[bool]] = {}
    hits = [r[f"{label}_hit"] for r in results]
    for r in results:
        cat = r.get("category", "unknown")
        by_cat.setdefault(cat, []).append(r[f"{label}_hit"])
    return {
        "hit_rate": round(sum(hits) / len(hits), 3),
        "n": len(hits),
        "by_category": {
            cat: {"hit_rate": round(sum(v) / len(v), 3), "n": len(v)}
            for cat, v in sorted(by_cat.items())
        },
    }


def main():
    args = parse_args()
    use_cuda = torch.cuda.is_available()
    device_map = "auto" if use_cuda else "cpu"
    print(f"Device: {'CUDA' if use_cuda else 'CPU'}")

    examples = load_test(args.test_data)
    print(f"Test examples: {len(examples)}")

    print("\n=== Evaluating BASE model ===")
    base_model, base_tok = _load_base_model(args.model, device_map)
    base_results = evaluate_model(base_model, base_tok, examples, "base", args)
    del base_model  # free VRAM before loading fine-tuned model
    if use_cuda:
        torch.cuda.empty_cache()

    adapter_path = args.adapter
    if not Path(adapter_path).exists():
        print(f"\n[WARN] Adapter not found at {adapter_path}. Skipping fine-tuned evaluation.")
        output = {
            "model": args.model,
            "base": summarise(base_results, "base"),
            "finetuned": None,
            "detail": base_results,
        }
    else:
        print(f"\n=== Evaluating FINE-TUNED model ({adapter_path}) ===")
        ft_model, ft_tok = _load_finetuned_model(args.model, adapter_path, device_map)
        ft_results = evaluate_model(ft_model, ft_tok, examples, "finetuned", args)

        # Merge per-example results
        merged = []
        for b, f in zip(base_results, ft_results):
            merged.append({**b,
                           "finetuned_answer": f["finetuned_answer"],
                           "finetuned_recall": f["finetuned_recall"],
                           "finetuned_hit": f["finetuned_hit"]})

        base_summary = summarise(base_results, "base")
        ft_summary = summarise(ft_results, "finetuned")

        print("\n=== Summary ===")
        print(f"Base     hit_rate: {base_summary['hit_rate']:.1%}")
        print(f"Finetuned hit_rate: {ft_summary['hit_rate']:.1%}")
        print(f"Delta:             {ft_summary['hit_rate'] - base_summary['hit_rate']:+.1%}")

        output = {
            "model": args.model,
            "adapter": adapter_path,
            "hit_threshold": HIT_THRESHOLD,
            "base": base_summary,
            "finetuned": ft_summary,
            "delta_hit_rate": round(ft_summary["hit_rate"] - base_summary["hit_rate"], 3),
            "detail": merged,
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
