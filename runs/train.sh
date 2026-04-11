#!/bin/bash
# =============================================================================
# Full training: nanochat + AttnRes v3 hybrid conv+attention (~978M params)
# Target: 1x RTX 5090 32GB
# Architecture: 24 conv + 8 attention layers (pattern SSSL)
# Usage:
#   tmux new -s train
#   bash runs/train.sh
# =============================================================================
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

# Env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY=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

# Model config — Hybrid Conv+AttnRes
DEPTH=26
ASPECT_RATIO=48
N_KV_HEAD=3
INTERMEDIATE_SIZE=4480
MLP_TYPE=swiglu
ROPE_BASE=1000000
ATTN_RES_BLOCK_SIZE=8
MAX_SEQ_LEN=2048
DEVICE_BATCH_SIZE=5
# 5 * 2048 = 10240, 522240 / 10240 = 51 exact
TOTAL_BATCH_SIZE=522240
WINDOW_PATTERN=SSSL
RUN_NAME="attnres-v3-hybrid-d${DEPTH}"

echo "============================================="
echo " nanochat + AttnRes v3 Hybrid (~978M params)"
echo " 24 conv + 8 attention (pattern: ${WINDOW_PATTERN})"
echo " depth=${DEPTH} | d_model=$((DEPTH * ASPECT_RATIO))"
echo " RTX 5090 | batch=${DEVICE_BATCH_SIZE}"
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
    --n-kv-head=${N_KV_HEAD} \
    --intermediate-size=${INTERMEDIATE_SIZE} \
    --mlp-type=${MLP_TYPE} \
    --rope-base=${ROPE_BASE} \
    --window-pattern=${WINDOW_PATTERN} \
    --max-seq-len=${MAX_SEQ_LEN} \
    --device-batch-size=${DEVICE_BATCH_SIZE} \
    --total-batch-size=${TOTAL_BATCH_SIZE} \
    --target-param-data-ratio=80 \
    --use-attn-res \
    --attn-res-block-size=${ATTN_RES_BLOCK_SIZE} \
    --run="${RUN_NAME}" \
    --save-every=10000
