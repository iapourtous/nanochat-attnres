#!/bin/bash
# =============================================================================
# DGX Spark setup (GB10 Blackwell, aarch64, CUDA 13, sm_121)
# Usage: bash runs/setup_server.sh
#
# Notes for DGX Spark:
#   - uv.lock pins a CPU wheel for torch (locked on x86). We nuke torch after
#     `uv sync` and reinstall from PyTorch nightly cu128 aarch64 wheels.
#   - sm_120 (in cu128 wheels) is binary-compatible with sm_121 (GB10).
#   - Unified 128GB memory: a training OOM can freeze the whole machine.
#     We disable swap to turn "brick" into "job dies cleanly".
#   - No Flash Attention 3 on Blackwell: SDPA fallback is actually faster.
# =============================================================================
set -e

echo "============================================="
echo " Server Setup: nanochat + AttnRes (DGX Spark)"
echo "============================================="

cd "$(dirname "$0")/.."

# Make sure uv is on PATH
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"

# 0. Safety: disable swap (if still on) to avoid swap-death-spiral OOMs
if [ "$(swapon --show)" != "" ]; then
    echo "[0/5] Swap is on -- disabling (recommended on DGX Spark)"
    echo "      (consider editing /etc/fstab to make this permanent)"
    sudo swapoff -a 2>/dev/null || echo "WARNING: could not disable swap (need sudo?). Continuing anyway."
fi

# 1. Base dependencies via uv sync (will install torch CPU on aarch64 -- we fix it in step 2)
echo ""
echo "[1/5] Installing base dependencies..."
uv sync --extra gpu

# 2. Replace torch CPU wheel with PyTorch nightly cu128 aarch64 (the only build
#    that actually works on DGX Spark GB10 today -- see https://github.com/natolambert/dgx-spark-setup).
#    sm_120 kernels in these wheels are binary-compatible with sm_121 (GB10).
echo ""
echo "[2/5] Installing PyTorch nightly cu128 aarch64 (GB10 Blackwell support)..."
uv pip uninstall torch torchvision torchaudio 2>/dev/null || true
uv pip install --pre --force-reinstall \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

echo ""
echo "  Verifying CUDA is available..."
CUDA_OK=$(uv run python -c "import torch; print('yes' if torch.cuda.is_available() else 'no')" 2>/dev/null || echo "no")
if [ "$CUDA_OK" != "yes" ]; then
    echo "  ERROR: PyTorch still does not see CUDA after nightly install."
    echo "         Debug: uv run python -c 'import torch; print(torch.__version__)'"
    exit 1
fi
uv run python -c "
import torch
print(f'  torch: {torch.__version__}')
print(f'  CUDA:  {torch.version.cuda}')
print(f'  GPU:   {torch.cuda.get_device_name(0)} | compute_cap: {torch.cuda.get_device_capability(0)}')
print(f'  VRAM:  {torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB (unified)')
"

# 3. Download all data (web + encyclopedic + books + arxiv + STEM)
echo ""
echo "[3/5] Downloading datasets..."
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi
uv run python -m nanochat.dataset \
  --fr -1 --en 170 \
  --wiki-fr -1 --wiki-en -1 \
  --books-fr -1 --diverse-fr -1 \
  --europarl -1 --arxiv -1 \
  -w 8

echo ""
echo "  Downloading STEM datasets (curriculum learning)..."
uv run python3 scripts/download_data.py \
  --nemmath -1 --owm -1 \
  --rcore -1 --synlog -1 \
  --docs -1 \
  -w 8

# 4. Train tokenizer on the full bilingual + STEM mix (deterministic shuffle)
echo ""
echo "[4/5] Training bilingual tokenizer..."
uv run python -m scripts.tok_train --max-chars 2000000000 --vocab-size 32768

# 5. Quick sanity check
echo ""
echo "[5/5] Sanity check (5 steps, no compile)..."
uv run python -m scripts.base_train \
  --depth=4 --max-seq-len=512 \
  --device-batch-size=4 --total-batch-size=2048 \
  --num-iterations=5 \
  --use-attn-res --attn-res-block-size=4 \
  --window-pattern=L --run=dummy --save-every=-1 \
  --no-compile

echo ""
echo "============================================="
echo " Setup complete! Launch training with:"
echo "   nohup bash runs/train.sh > train.log 2>&1 &"
echo "   tail -f train.log"
echo "============================================="
