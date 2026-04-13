#!/bin/bash
# =============================================================================
# H100 SXM setup (Hopper sm_90, 80GB HBM3, x86_64, CUDA 12.8)
# Usage: bash runs/setup_server.sh
# =============================================================================
set -e

echo "============================================="
echo " Server Setup: nanochat + AttnRes (H100)"
echo "============================================="

cd "$(dirname "$0")/.."

# Make sure uv is on PATH
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"

# 1. Install dependencies (cu128 x86_64 wheel works out-of-the-box on Hopper)
echo ""
echo "[1/5] Installing dependencies..."
uv sync --extra gpu
uv run --no-sync python -c "
import torch
print(f'  torch: {torch.__version__}')
print(f'  CUDA:  {torch.version.cuda}')
print(f'  GPU:   {torch.cuda.get_device_name(0)} | compute_cap: {torch.cuda.get_device_capability(0)}')
print(f'  VRAM:  {torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB HBM3')
"

# 2. Download all data (web + encyclopedic + books + arxiv)
echo ""
echo "[2/5] Downloading datasets..."
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi
uv run --no-sync python -m nanochat.dataset \
  --fr -1 --en 170 \
  --wiki-fr -1 --wiki-en -1 \
  --books-fr -1 --diverse-fr -1 \
  --europarl -1 --arxiv -1 \
  -w 8

# 3. Download STEM datasets (curriculum learning)
echo ""
echo "[3/5] Downloading STEM datasets..."
uv run --no-sync python scripts/download_data.py \
  --nemmath -1 --owm -1 \
  --rcore -1 --synlog -1 \
  --docs -1 \
  -w 8

# 4. Train tokenizer on the full bilingual + STEM mix (deterministic shuffle)
echo ""
echo "[4/5] Training bilingual tokenizer..."
uv run --no-sync python -m scripts.tok_train --max-chars 2000000000 --vocab-size 32768

# 5. Quick sanity check (same architecture as real training, miniature scale)
echo ""
echo "[5/5] Sanity check (5 steps, hybrid SSSL + FP8 + compile-off)..."
uv run --no-sync python -m scripts.base_train \
  --depth=8 --max-seq-len=512 \
  --mlp-type=swiglu \
  --window-pattern=SSSL \
  --device-batch-size=4 --total-batch-size=2048 \
  --num-iterations=5 \
  --use-attn-res --attn-res-block-size=8 \
  --curriculum \
  --fp8 --fp8-recipe=tensorwise \
  --run=dummy --save-every=-1 \
  --no-compile

echo ""
echo "============================================="
echo " Setup complete! Launch training with:"
echo "   nohup bash runs/train.sh > train.log 2>&1 &"
echo "   tail -f train.log"
echo "============================================="
