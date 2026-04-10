#!/bin/bash
# =============================================================================
# Full training: nanochat + AttnRes v3 — GQA + SwiGLU (~950M params)
# Target: 1x H100 80GB
# Usage:
#   tmux new -s train
#   bash runs/train_v3.sh
# =============================================================================
set -e

cd "$(dirname "$0")/.."
source .venv/bin/activate

# Wandb
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_API_KEY=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

# Model config — AttnRes v3: GQA 4:1 + SwiGLU + RoPE 1M
DEPTH=32
ASPECT_RATIO=48
# d_model = 32 * 48 = 1536
# n_head = 1536 / 128 = 12
# n_kv_head = 3 (GQA 4:1)
# SwiGLU intermediate = 4480
# Total: ~950M params
N_KV_HEAD=3
INTERMEDIATE_SIZE=4480
MLP_TYPE=swiglu
ROPE_BASE=1000000
ATTN_RES_BLOCK_SIZE=8
MAX_SEQ_LEN=2048
DEVICE_BATCH_SIZE=8
TOTAL_BATCH_SIZE=524288
RUN_NAME="attnres-v3-d${DEPTH}"

echo "============================================="
echo " nanochat + AttnRes v3 (~950M params)"
echo " GQA 4:1 + SwiGLU + RoPE 1M"
echo " depth=${DEPTH} | d_model=$((DEPTH * ASPECT_RATIO))"
echo " batch=${DEVICE_BATCH_SIZE} | total=${TOTAL_BATCH_SIZE}"
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
    --fp8 \
    --depth=${DEPTH} \
    --aspect-ratio=${ASPECT_RATIO} \
    --n-kv-head=${N_KV_HEAD} \
    --intermediate-size=${INTERMEDIATE_SIZE} \
    --mlp-type=${MLP_TYPE} \
    --rope-base=${ROPE_BASE} \
    --max-seq-len=${MAX_SEQ_LEN} \
    --device-batch-size=${DEVICE_BATCH_SIZE} \
    --total-batch-size=${TOTAL_BATCH_SIZE} \
    --target-param-data-ratio=75 \
    --use-attn-res \
    --attn-res-block-size=${ATTN_RES_BLOCK_SIZE} \
    --run="${RUN_NAME}" \
    --save-every=2000
