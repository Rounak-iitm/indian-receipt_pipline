"""
Merges a LoRA adapter into the base model, then converts to GGUF and
quantizes for fast local CPU/GPU inference (via llama.cpp).

This script assumes you have llama.cpp cloned alongside this repo:
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp && pip install -r requirements.txt

Usage:
    python scripts/quantize_export.py --model models/qlora-finetuned --bits 4 \
        --llama-cpp-path ../llama.cpp
"""
import argparse
import subprocess
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

BITS_TO_GGUF_TYPE = {
    2: "Q2_K",
    4: "Q4_K_M",
    5: "Q5_K_M",
    8: "Q8_0",
}


def merge_lora(adapter_path: str, merged_out: str):
    print(f"Loading adapter from {adapter_path} and merging into base model...")
    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_path, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    merged = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    Path(merged_out).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_out)
    tokenizer.save_pretrained(merged_out)
    print(f"Merged full-precision model saved to {merged_out}")


def convert_and_quantize(merged_dir: str, llama_cpp_path: str, bits: int, out_dir: str):
    gguf_type = BITS_TO_GGUF_TYPE[bits]
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    f16_gguf = str(Path(out_dir) / "model-f16.gguf")
    quant_gguf = str(Path(out_dir) / f"model-{gguf_type}.gguf")

    print("Converting HF model to GGUF (f16)...")
    subprocess.run(
        [
            "python", str(Path(llama_cpp_path) / "convert_hf_to_gguf.py"),
            merged_dir, "--outfile", f16_gguf, "--outtype", "f16",
        ],
        check=True,
    )

    print(f"Quantizing to {gguf_type}...")
    quantize_bin = Path(llama_cpp_path) / "llama-quantize"
    subprocess.run([str(quantize_bin), f16_gguf, quant_gguf, gguf_type], check=True)

    print(f"Done. Quantized model at: {quant_gguf}")
    return quant_gguf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="path to LoRA adapter dir")
    ap.add_argument("--bits", type=int, default=4, choices=[2, 4, 5, 8])
    ap.add_argument("--llama-cpp-path", type=str, default="../llama.cpp")
    ap.add_argument("--out", type=str, default="models/quantized")
    args = ap.parse_args()

    merged_dir = str(Path(args.model).parent / (Path(args.model).name + "-merged"))
    merge_lora(args.model, merged_dir)
    convert_and_quantize(merged_dir, args.llama_cpp_path, args.bits, args.out)


if __name__ == "__main__":
    main()
