#!/bin/bash
# =============================================================================
# Full training: nanochat + AttnRes v2 — 1x H100 80GB
# Usage:
#   tmux new -s train
#   bash runs/train_h100.sh
# =============================================================================
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

# Wandb
export WANDB_API_KEY=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

# Model config
DEPTH=32
ASPECT_RATIO=39
ATTN_RES_BLOCK_SIZE=8
MAX_SEQ_LEN=2048
DEVICE_BATCH_SIZE=16
RUN_NAME="attnres-v2-d${DEPTH}"

echo "============================================="
echo " nanochat + AttnRes v2 (~949M params)"
echo " depth=${DEPTH} | d_model=$((DEPTH * ASPECT_RATIO))"
echo " H100 single GPU | batch=${DEVICE_BATCH_SIZE}"
echo "============================================="

# Check data
DATA_DIR="$HOME/.cache/nanochat/base_data_bilingual"
SHARD_COUNT=$(ls "$DATA_DIR"/*.parquet 2>/dev/null | wc -l)
echo "Dataset: ${SHARD_COUNT} shards"

# Check tokenizer
if [ ! -f "$HOME/.cache/nanochat/tokenizer/tokenizer.pkl" ]; then
    echo "ERROR: No tokenizer found. Run setup_server.sh first."
    exit 1
fi

echo "Starting training..."
echo ""

python -m scripts.base_train \
    --depth=${DEPTH} \
    --aspect-ratio=${ASPECT_RATIO} \
    --max-seq-len=${MAX_SEQ_LEN} \
    --device-batch-size=${DEVICE_BATCH_SIZE} \
    --total-batch-size=524288 \
    --target-param-data-ratio=75 \
    --use-attn-res \
    --attn-res-block-size=${ATTN_RES_BLOCK_SIZE} \
    --run="${RUN_NAME}" \
    --save-every=2000
