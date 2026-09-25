"""
Evaluate base vs. fine-tuned Qwen2.5-0.5B on tool-selection.

Metrics:
  exact_tool_match    -- correct tool name(s), regardless of args
  exact_arg_match     -- correct tool name(s) AND all expected args match
  json_valid          -- output is parseable JSON
  p50_latency_ms      -- median per-example generation latency

All metrics reported with bootstrap 95% CIs (10 000 resamples, n is small).

Evaluation sources:
  1. data/finetune/tool_test.jsonl           -- held-out templates
  2. data/evaluation/agent_questions_v2.json -- real agent questions with expected_tools

Run:
  python -m scripts.evaluate_tool_selection [--adapter models/tool_selection_adapter/adapter]
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

ROOT = Path(__file__).parent.parent
DEFAULT_ADAPTER = ROOT / "models/tool_selection_adapter/adapter"
DEFAULT_TEST = ROOT / "data/finetune/tool_test.jsonl"
AGENT_QS = ROOT / "data/evaluation/agent_questions_v2.json"
DEFAULT_OUT = ROOT / "data/evaluation/tool_selection_results.json"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_BOOTSTRAP = 10_000


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    ap.add_argument("--test-data", default=str(DEFAULT_TEST))
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-length", type=int, default=512)
    return ap.parse_args()


# ── Parsing helpers ──────────────────────────────────────────────────────────

def _parse_tool_call(text: str) -> dict | list | None:
    """Return parsed JSON tool call, or None if invalid."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def _normalise(obj: dict | list | None) -> list[dict]:
    """Normalise to a flat list of {tool, arguments} dicts."""
    if obj is None:
        return []
    if isinstance(obj, dict):
        return [obj]
    if isinstance(obj, list):
        return obj
    return []


def _tools_from_normalised(calls: list[dict]) -> list[str]:
    return [c.get("tool", c.get("name", "")) for c in calls]


def _args_from_normalised(calls: list[dict], tool: str) -> dict:
    for c in calls:
        if c.get("tool", c.get("name")) == tool:
            return c.get("arguments", c.get("input", {})) or {}
    return {}


# ── Bootstrap CI ────────────────────────────────────────────────────────────

def bootstrap_ci(values: list[float], n: int = N_BOOTSTRAP, seed: int = 0) -> tuple[float, float]:
    """Return (lower, upper) 95% CI for the mean via bootstrap."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    k = len(values)
    means = sorted(
        sum(rng.choices(values, k=k)) / k
        for _ in range(n)
    )
    lo = means[int(0.025 * n)]
    hi = means[int(0.975 * n)]
    return (round(lo, 4), round(hi, 4))


# ── Load examples ────────────────────────────────────────────────────────────

def load_template_test(path: str) -> list[dict]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def load_agent_questions(path: str) -> list[dict]:
    """Convert agent_questions_v2 rows with expected_tools into eval examples."""
    try:
        qs = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    examples = []
    for q in qs:
        tools = q.get("expected_tools") or []
        if not tools:
            continue
        args_map = q.get("expected_args") or {}
        calls = [{"tool": t, "arguments": args_map.get(t, {})} for t in tools]
        from scripts.prepare_tool_selection_data import SYSTEM_PROMPT, _answer_json
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q["question"]},
                {"role": "assistant", "content": _answer_json(calls)},
            ],
            "meta": {
                "template_id": f"agent.{q['id']}",
                "tools": tools,
                "source": "agent_questions_v2",
                "q_id": q["id"],
            },
        })
    return examples


# ── Model helpers ────────────────────────────────────────────────────────────

def _load_base(model_id: str, device_map: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map=device_map,
        trust_remote_code=True, dtype=torch.bfloat16,
    )
    return model, tok


def _load_finetuned(model_id: str, adapter_path: str, device_map: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map=device_map,
        trust_remote_code=True, dtype=torch.bfloat16,
    )
    return PeftModel.from_pretrained(base, adapter_path), tok


def generate(model, tokenizer, messages: list[dict], max_new_tokens: int, max_length: int) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=1.0,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── Evaluation loop ──────────────────────────────────────────────────────────

def evaluate(model, tokenizer, examples: list[dict], label: str, args) -> list[dict]:
    results = []
    for i, ex in enumerate(examples, 1):
        messages = ex["messages"]
        ref_answer = next(m["content"] for m in messages if m["role"] == "assistant")
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        ref_calls = _normalise(_parse_tool_call(ref_answer))
        ref_tools = _tools_from_normalised(ref_calls)

        t0 = time.monotonic()
        generated = generate(model, tokenizer, prompt_messages, args.max_new_tokens, args.max_length)
        latency_ms = round((time.monotonic() - t0) * 1000, 1)

        parsed = _parse_tool_call(generated)
        gen_calls = _normalise(parsed)
        gen_tools = _tools_from_normalised(gen_calls)

        json_valid = parsed is not None
        exact_tool = sorted(gen_tools) == sorted(ref_tools)

        # Arg match: for each expected tool, check expected args are a subset of generated args
        arg_match = exact_tool
        if arg_match:
            for ref_c in ref_calls:
                t_name = ref_c.get("tool", ref_c.get("name", ""))
                ref_args = ref_c.get("arguments", {}) or {}
                if not ref_args:
                    continue  # no expected args to verify
                gen_args = _args_from_normalised(gen_calls, t_name)
                for k, v in ref_args.items():
                    if gen_args.get(k) != v:
                        arg_match = False
                        break
                if not arg_match:
                    break

        status = ("TOOL+ARG+" if arg_match else "TOOL+ARG-" if exact_tool else "TOOL-")
        source = ex.get("meta", {}).get("source", "template")
        print(f"  [{i:>3}/{len(examples)}] {status}  json={int(json_valid)}  "
              f"{source[:8]}  {ex.get('meta',{}).get('template_id','')[:30]}")

        results.append({
            "template_id": ex.get("meta", {}).get("template_id"),
            "source": source,
            "question": next(m["content"] for m in messages if m["role"] == "user"),
            "reference": ref_answer,
            f"{label}_generated": generated,
            f"{label}_json_valid": json_valid,
            f"{label}_exact_tool": exact_tool,
            f"{label}_arg_match": arg_match,
            f"{label}_latency_ms": latency_ms,
        })
    return results


def _summary(results: list[dict], label: str) -> dict:
    def _rate(key: str) -> float:
        vals = [r[key] for r in results]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    json_vals = [float(r[f"{label}_json_valid"]) for r in results]
    tool_vals = [float(r[f"{label}_exact_tool"]) for r in results]
    arg_vals = [float(r[f"{label}_arg_match"]) for r in results]
    lat_vals = sorted(r[f"{label}_latency_ms"] for r in results)
    p50 = lat_vals[len(lat_vals) // 2] if lat_vals else 0.0

    return {
        "n": len(results),
        "json_valid": _rate(f"{label}_json_valid"),
        "json_valid_ci95": bootstrap_ci(json_vals),
        "exact_tool_match": _rate(f"{label}_exact_tool"),
        "exact_tool_ci95": bootstrap_ci(tool_vals),
        "arg_match": _rate(f"{label}_arg_match"),
        "arg_match_ci95": bootstrap_ci(arg_vals),
        "p50_latency_ms": p50,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    use_cuda = torch.cuda.is_available()
    device_map = "auto" if use_cuda else "cpu"
    print(f"Device: {'CUDA' if use_cuda else 'CPU'}")

    template_test = load_template_test(args.test_data)
    agent_test = load_agent_questions(AGENT_QS)
    all_examples = template_test + agent_test
    print(f"Test examples: {len(all_examples)} "
          f"({len(template_test)} templates + {len(agent_test)} agent traces)")

    print("\n=== BASE model ===")
    base_model, base_tok = _load_base(args.model, device_map)
    base_results = evaluate(base_model, base_tok, all_examples, "base", args)
    del base_model
    if use_cuda:
        torch.cuda.empty_cache()

    adapter_path = args.adapter
    if not Path(adapter_path).exists():
        print(f"\n[WARN] Adapter not found at {adapter_path}. Skipping fine-tuned eval.")
        output = {
            "model": args.model,
            "base": _summary(base_results, "base"),
            "finetuned": None,
            "detail": base_results,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"\nWrote {args.output}")
    else:
        print(f"\n=== FINE-TUNED model ({adapter_path}) ===")
        ft_model, ft_tok = _load_finetuned(args.model, adapter_path, device_map)
        ft_results = evaluate(ft_model, ft_tok, all_examples, "finetuned", args)

        merged = []
        for b, f in zip(base_results, ft_results):
            merged.append({**b,
                           "finetuned_generated": f["finetuned_generated"],
                           "finetuned_json_valid": f["finetuned_json_valid"],
                           "finetuned_exact_tool": f["finetuned_exact_tool"],
                           "finetuned_arg_match": f["finetuned_arg_match"],
                           "finetuned_latency_ms": f["finetuned_latency_ms"]})

        base_s = _summary(base_results, "base")
        ft_s = _summary(ft_results, "finetuned")

        output = {
            "model": args.model,
            "adapter": adapter_path,
            "base": base_s,
            "finetuned": ft_s,
            "delta": {
                "json_valid": round(ft_s["json_valid"] - base_s["json_valid"], 3),
                "exact_tool_match": round(ft_s["exact_tool_match"] - base_s["exact_tool_match"], 3),
                "arg_match": round(ft_s["arg_match"] - base_s["arg_match"], 3),
            },
            "detail": merged,
        }

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"\nWrote {args.output}")

        print("\n=== Summary ===")
        for metric in ("json_valid", "exact_tool_match", "arg_match"):
            b_val = base_s[metric]
            f_val = ft_s[metric]
            b_ci = base_s.get(f"{metric}_ci95", ("?", "?"))
            f_ci = ft_s.get(f"{metric.replace('_match','')}_ci95", ft_s.get(f"{metric}_ci95", ("?", "?")))
            print(f"  {metric:<20} base={b_val:.1%} {b_ci}  ft={f_val:.1%} {f_ci}  delta={f_val-b_val:+.1%}")
        print(f"  {'p50_latency_ms':<20} base={base_s['p50_latency_ms']:.0f}ms  "
              f"ft={ft_s['p50_latency_ms']:.0f}ms")


if __name__ == "__main__":
    main()
