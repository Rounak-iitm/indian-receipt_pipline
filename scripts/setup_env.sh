#!/usr/bin/env bash
# Sets up a CUDA-enabled PyTorch environment for the RTX 3050 (4GB).
# Run this on YOUR laptop, not in a sandbox without a GPU.
set -e

echo "=== Indian Receipt AI: environment setup ==="

# 1. (Recommended) create a virtual environment first:
#    python -m venv .venv && source .venv/bin/activate   (Linux/Mac)
#    python -m venv .venv && .venv\Scripts\activate       (Windows)

# 2. Install CUDA-enabled torch.
#    Check your installed CUDA driver version with `nvidia-smi` first (top-right
#    corner shows "CUDA Version: X.Y" - that's the MAX version your driver supports).
#    cu121 works with driver CUDA 12.1+; use cu118 if your driver is older.
echo "Installing PyTorch (CUDA 12.1 build)..."
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Install the rest of the requirements
echo "Installing project requirements..."
pip install -r requirements.txt

# 4. Sanity check
python scripts/check_gpu.py

echo "=== Done. If check_gpu.py printed your RTX 3050, you're ready. ==="
