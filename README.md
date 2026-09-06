# Indian Receipt & Invoice AI

Fine-tuning and quantizing a small language model (Qwen2.5-1.5B-Instruct) to turn
Indian receipts, invoices, and bills into structured JSON — trained and run
locally on a 4GB laptop GPU (RTX 3050).

## Pipeline

```
Receipt image/text
      │
      ▼
   OCR (Tesseract / EasyOCR)
      │
      ▼
  Raw noisy text
      │
      ▼
  Fine-tuned Qwen2.5-1.5B (QLoRA)
      │
      ▼
  Structured JSON (vendor, items, GST, total, payment, date)
      │
      ▼
  Validation layer (schema + arithmetic checks)
      │
      ▼
  Streamlit demo app
```

## Repo layout

```
indian-receipt-ai/
├── data/
│   ├── raw/            # real receipt images/text you collect (optional)
│   └── synthetic/       # generated synthetic receipts + labels
├── scripts/
│   ├── setup_env.sh              # installs CUDA-enabled torch + deps
│   ├── check_gpu.py              # verifies RTX 3050 is visible to torch
│   ├── generate_synthetic_data.py# builds the synthetic receipt dataset
│   ├── baseline_eval.py          # zero-shot Qwen2.5-1.5B accuracy
│   ├── train_qlora.py            # QLoRA fine-tuning, sized for 4GB VRAM
│   ├── quantize_export.py        # export fine-tuned model to GGUF
│   └── eval_compare.py           # base vs LoRA vs QLoRA vs quantized
├── models/               # local checkpoints (gitignored, large files)
├── app/
│   └── streamlit_app.py  # upload receipt -> OCR -> extraction -> JSON
└── notebooks/            # exploration / experiment notebooks
```

## Quickstart (on your laptop, not this sandbox)

```bash
# 1. Set up environment (installs CUDA torch, bitsandbytes, peft, transformers)
bash scripts/setup_env.sh

# 2. Verify GPU is visible
python scripts/check_gpu.py

# 3. Generate synthetic training data
python scripts/generate_synthetic_data.py --n 10000 --out data/synthetic

# 4. Baseline: how good is the un-tuned model?
python scripts/baseline_eval.py --model Qwen/Qwen2.5-1.5B-Instruct --data data/synthetic/test.jsonl

# 5. Fine-tune with QLoRA (4-bit, fits in 4GB VRAM)
python scripts/train_qlora.py --config configs/qlora_1_5b.yaml

# 6. Quantize the fine-tuned model to GGUF for fast local inference
python scripts/quantize_export.py --model models/qlora-finetuned --bits 4

# 7. Compare all variants
python scripts/eval_compare.py

# 8. Run the demo
streamlit run app/streamlit_app.py
```

## Research question

> How much can a 1.5B-parameter language model be specialized for Indian
> financial-document extraction using parameter-efficient fine-tuning (QLoRA)
> and post-training quantization, while running entirely on a 4GB consumer GPU?

## Status
- [x] Repo scaffolded
- [ ] Environment verified on RTX 3050
- [ ] Synthetic dataset generated
- [ ] Baseline eval run
- [ ] QLoRA fine-tuning complete
- [ ] Quantized + benchmarked
- [ ] Streamlit demo working
