
import torch

print(f"torch version: {torch.__version__}")
cuda_ok = torch.cuda.is_available()
print(f"CUDA available: {cuda_ok}")

if cuda_ok:
    name = torch.cuda.get_device_name(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU: {name}")
    print(f"Total VRAM: {total_mem_gb:.2f} GB")
    if total_mem_gb < 4.5:
        print(
            "\nNote: you're on a 4GB-class GPU. For fine-tuning, stick to:\n"
            "  - 4-bit QLoRA (not full fine-tuning, not plain LoRA in fp16)\n"
            "  - Qwen2.5-0.5B or 1.5B (skip anything 3B+)\n"
            "  - small batch size (1-2) with gradient accumulation\n"
            "  - gradient checkpointing enabled\n"
        )
else:
    print(
        "\nCUDA is NOT available. Common causes:\n"
        "  1. You installed the CPU-only torch build (torch==X.Y.Z+cpu).\n"
        "     Fix: pip uninstall torch && pip install torch --index-url "
        "https://download.pytorch.org/whl/cu121\n"
        "  2. NVIDIA drivers aren't installed/up to date. Run `nvidia-smi` to check.\n"
        "  3. You're on WSL without GPU passthrough configured.\n"
    )
