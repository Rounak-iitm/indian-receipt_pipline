"""
Runs evaluation across model variants (base, LoRA fine-tuned, quantized) and
prints/saves a comparison table like the one in the project README.

Usage:
    python scripts/eval_compare.py
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

VARIANTS = [
    {"label": "Base (Qwen2.5-1.5B)", "model": "Qwen/Qwen2.5-1.5B-Instruct", "adapter": False},
    {"label": "QLoRA fine-tuned", "model": "models/qlora-finetuned", "adapter": True},
    # Add a row here once you've exported a quantized GGUF and have a way to
    # run inference on it (e.g. via llama-cpp-python) - GGUF eval needs a
    # separate inference path since baseline_eval.py uses transformers.generate().
]


def run_variant(v: dict, data_path: str, limit: int) -> str:
    cmd = [
        sys.executable, "scripts/baseline_eval.py",
        "--model", v["model"],
        "--data", data_path,
        "--limit", str(limit),
    ]
    if v["adapter"]:
        cmd.append("--adapter")
    print(f"\n=== Running: {v['label']} ===")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout[-2000:])
    if result.returncode != 0:
        print(f"FAILED: {result.stderr[-2000:]}", file=sys.stderr)
    return result.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data/synthetic/test.jsonl")
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()

    outputs = {}
    for v in VARIANTS:
        if v["adapter"] and not Path(v["model"]).exists():
            print(f"Skipping {v['label']}: {v['model']} not found (train it first).")
            continue
        outputs[v["label"]] = run_variant(v, args.data, args.limit)

    print("\n\n=== Summary ===")
    for label, out in outputs.items():
        print(f"\n--- {label} ---")
        print(out.split("=== Results")[-1] if "=== Results" in out else out)


if __name__ == "__main__":
    main()
