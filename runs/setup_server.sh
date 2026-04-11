#!/bin/bash
# =============================================================================
# Server setup — run ONCE on the remote machine (single GPU)
# Usage: bash runs/setup_server.sh
# =============================================================================
set -e

echo "============================================="
echo " Server Setup: nanochat + AttnRes (1 GPU)"
echo "============================================="

cd "$(dirname "$0")/.."

# 1. Install dependencies
echo "[1/4] Installing dependencies..."
uv sync --extra gpu
uv run python -c "import torch; print(f'     GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB')"

# 2. Download all data
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
  --nemmath 350 --owm 150 \
  --rcore -1 --synlog -1 \
  --docs -1 \
  -w 8

# 3. Train tokenizer
echo ""
echo "[3/4] Training bilingual tokenizer..."
uv run python -m scripts.tok_train --max-chars 2000000000 --vocab-size 32768

# 4. Quick sanity check
echo ""
echo "[4/4] Sanity check (5 steps)..."
uv run python -m scripts.base_train \
  --depth=4 --max-seq-len=512 \
  --device-batch-size=4 --total-batch-size=2048 \
  --num-iterations=5 \
  --use-attn-res --attn-res-block-size=4 \
  --window-pattern=L --run=dummy --save-every=-1

echo ""
echo "============================================="
echo " Setup complete! Launch training with:"
echo "   nohup bash runs/train_attnres.sh > train.log 2>&1 &"
echo "   tail -f train.log"
echo "============================================="
