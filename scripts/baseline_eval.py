"""
Evaluates a model (base or fine-tuned) on the receipt extraction test set.
Scores: JSON validity rate, per-field exact-match accuracy, total-amount
exact-match, and end-to-end latency.

Usage:
    python scripts/baseline_eval.py --model Qwen/Qwen2.5-1.5B-Instruct --data data/synthetic/test.jsonl
    python scripts/baseline_eval.py --model models/qlora-finetuned --data data/synthetic/test.jsonl --adapter
"""
import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

FIELDS_TO_SCORE = ["vendor", "total", "gst", "payment_method", "date", "currency"]


def load_model(model_path: str, is_adapter: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if is_adapter:
        from peft import AutoPeftModelForCausalLM
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path, device_map="auto", torch_dtype=torch.bfloat16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", torch_dtype=torch.bfloat16
        )
    model.eval()
    return model, tokenizer


def run_inference(model, tokenizer, instruction: str, max_new_tokens: int = 400) -> tuple[str, float]:
    messages = [
        {"role": "system", "content": "You extract structured data from Indian receipts and respond with JSON only."},
        {"role": "user", "content": instruction},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    start = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = time.time() - start

    gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return gen, latency


def try_parse_json(text: str):
    # Models sometimes wrap JSON in markdown fences or add stray text.
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1]
        text = text.replace("json", "", 1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def score_record(pred: dict, gold: dict) -> dict:
    result = {}
    for field in FIELDS_TO_SCORE:
        if pred is None:
            result[field] = False
            continue
        pv, gv = pred.get(field), gold.get(field)
        if isinstance(gv, float):
            result[field] = pv is not None and abs(float(pv) - gv) < 0.01
        else:
            result[field] = str(pv).strip().lower() == str(gv).strip().lower() if pv is not None else False
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--adapter", action="store_true", help="model path is a PEFT/LoRA adapter dir")
    ap.add_argument("--limit", type=int, default=200, help="max number of test examples to evaluate")
    args = ap.parse_args()

    model, tokenizer = load_model(args.model, args.adapter)

    records = []
    with open(args.data) as f:
        for line in f:
            records.append(json.loads(line))
    records = records[: args.limit]

    valid_json = 0
    field_correct = {f: 0 for f in FIELDS_TO_SCORE}
    latencies = []

    for i, rec in enumerate(records):
        gen, latency = run_inference(model, tokenizer, rec["instruction"])
        latencies.append(latency)
        pred = try_parse_json(gen)
        if pred is not None:
            valid_json += 1
        scores = score_record(pred, rec["label"])
        for f, ok in scores.items():
            field_correct[f] += int(ok)

        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{len(records)}")

    n = len(records)
    print(f"\n=== Results for {args.model} (n={n}) ===")
    print(f"Valid JSON rate: {valid_json / n:.1%}")
    for f in FIELDS_TO_SCORE:
        print(f"  {f:15s}: {field_correct[f] / n:.1%}")
    print(f"Avg latency: {sum(latencies) / n:.2f}s")


if __name__ == "__main__":
    main()
