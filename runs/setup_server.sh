#!/bin/bash
# =============================================================================
# DGX Spark setup (GB10 Blackwell, aarch64, CUDA 13, sm_121)
# Usage: bash runs/setup_server.sh
#
# Notes for DGX Spark:
#   - uv.lock pins a CPU wheel for torch because it was generated on x86,
#     so we force-reinstall the aarch64 cu128 wheel after `uv sync`.
#   - sm_120 (in cu128 wheels) is binary-compatible with sm_121 (GB10).
#   - Unified 128GB memory: a training OOM can freeze the whole machine.
#     Disable swap to turn "brick" into "job dies cleanly".
#   - No Flash Attention 3 on Blackwell: SDPA fallback is actually faster.
# =============================================================================
set -e

echo "============================================="
echo " Server Setup: nanochat + AttnRes (DGX Spark)"
echo "============================================="

cd "$(dirname "$0")/.."

# 0. Safety: disable swap (if still on) to avoid swap-death-spiral OOMs
if [ "$(swapon --show)" != "" ]; then
    echo "[0/4] Swap is on -- disabling (recommended on DGX Spark)"
    echo "      (consider editing /etc/fstab to make this permanent)"
    sudo swapoff -a || echo "WARNING: could not disable swap (need sudo?). Continuing anyway."
fi

# 1. Install dependencies
echo "[1/4] Installing dependencies..."
uv sync --extra gpu

# DGX Spark fix: uv.lock pins a CPU torch wheel (locked on x86).
# Force-install the aarch64 cu128 wheel so CUDA is actually available.
CUDA_OK=$(uv run python -c "import torch; print('yes' if torch.cuda.is_available() else 'no')" 2>/dev/null || echo "no")
if [ "$CUDA_OK" != "yes" ]; then
    echo "  Torch CPU wheel detected. Force-reinstalling cu128 aarch64 wheel..."
    uv pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1 --force-reinstall
fi
uv run python -c "import torch; print(f'     GPU: {torch.cuda.get_device_name(0)} | compute_cap: {torch.cuda.get_device_capability(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB')"

# 2. Download all data (web + encyclopedic + books + arxiv)
echo ""
echo "[2/4] Downloading datasets..."
uv run python -m nanochat.dataset \
  --fr -1 --en 170 \
  --wiki-fr -1 --wiki-en -1 \
  --books-fr -1 --diverse-fr -1 \
  --europarl -1 --arxiv -1 \
  -w 8

# 2b. Download STEM datasets (curriculum learning)
echo ""
echo "[2b/4] Downloading STEM datasets..."
if [ -f .env ]; then
    export $(grep HF_TOKEN .env | xargs)
fi
uv run python3 scripts/download_data.py \
  --nemmath -1 --owm -1 \
  --rcore -1 --synlog -1 \
  --docs -1 \
  -w 8

# 3. Train tokenizer on the full bilingual + STEM mix (deterministic shuffle)
echo ""
echo "[3/4] Training bilingual tokenizer..."
uv run python -m scripts.tok_train --max-chars 2000000000 --vocab-size 32768

# 4. Quick sanity check (SDPA fallback on Blackwell, no compile on ARM for speed of first run)
echo ""
echo "[4/4] Sanity check (5 steps)..."
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
