#!/bin/bash
# =============================================================================
# Full training: nanochat + AttnRes on single H200
# Usage:
#   nohup bash runs/train_attnres.sh > train.log 2>&1 &
#   tail -f train.log
# =============================================================================
set -e

DEPTH=32
ASPECT_RATIO=48
ATTN_RES_BLOCK_SIZE=8
DEVICE_BATCH_SIZE=32
MAX_SEQ_LEN=2048
RUN_NAME="attnres-bilingual-d${DEPTH}"

echo "============================================="
echo " nanochat + AttnRes Training (single GPU)"
echo " depth=${DEPTH} | ar=${ASPECT_RATIO} | d_model=$((DEPTH * ASPECT_RATIO))"
echo "============================================="

cd /workspace/nanochat-attnres

# Check data
DATA_DIR="$HOME/.cache/nanochat/base_data_bilingual"
SHARD_COUNT=$(ls $DATA_DIR/*.parquet 2>/dev/null | wc -l)
echo "Dataset: ${SHARD_COUNT} shards"

# Check tokenizer
if [ ! -f "$HOME/.cache/nanochat/tokenizer/tokenizer.pkl" ]; then
    echo "ERROR: Run setup_server.sh first"
    exit 1
fi

echo "Starting training..."
echo ""

uv run python -m scripts.base_train \
    --depth=${DEPTH} \
    --aspect-ratio=${ASPECT_RATIO} \
    --max-seq-len=${MAX_SEQ_LEN} \
    --device-batch-size=${DEVICE_BATCH_SIZE} \
    --use-attn-res \
    --attn-res-block-size=${ATTN_RES_BLOCK_SIZE} \
    --run="${RUN_NAME}" \
    --save-every=2000
