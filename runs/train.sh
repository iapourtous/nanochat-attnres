#!/bin/bash
# =============================================================================
# Full training: nanochat + AttnRes (pure attention, ~750M params)
# Target: 1x NVIDIA H100 SXM (80GB HBM3, Hopper sm_90, x86_64)
# Architecture: 26 attention layers (pattern L -- no GatedConv, back to the
#               original nanochat + AttnRes recipe that was proven in the paper)
# Usage:
#   tmux new -s train
#   bash runs/train.sh
#
# H100 notes:
#   - Flash Attention 3 (FA3) is NATIVE on Hopper -- all 26 layers benefit.
#   - FP8 (E4M3 fwd, E5M2 bwd) via torch._scaled_mm -- H100 has FP8 tensor cores,
#     gives ~1.5-2x speedup over BF16.
#   - HBM3 bandwidth 3350 GB/s, compute-bound with FA3.
#   - torch.compile enabled: first compile 10-30 min, then ~30% speedup.
# =============================================================================
set -e

cd "$(dirname "$0")/.."
# Make sure uv is on PATH
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"
source .venv/bin/activate

# Env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Model config — Pure attention + AttnRes (depth 26, ~750M params)
# Rollback from the GatedConv hybrid (samples were degrading at ~6K steps) back
# to the original nanochat + AttnRes recipe that worked.
DEPTH=26
ASPECT_RATIO=48
# 26 * 48 = 1248 → nudged to 1280 (next multiple of head_dim=128) → n_head=10
# n_kv_head=3 gets auto-clamped to 2 (to divide num_heads=10 evenly)
N_KV_HEAD=3
INTERMEDIATE_SIZE=4480
MLP_TYPE=swiglu
ROPE_BASE=1000000
ATTN_RES_BLOCK_SIZE=8
MAX_SEQ_LEN=2048
# H100 80GB HBM3. Smaller model (~750M) lets us bump the micro-batch.
# 16 * 2048 = 32768 tokens/micro-batch, 524288/32768 = 16 grad_accum exact.
DEVICE_BATCH_SIZE=16
TOTAL_BATCH_SIZE=524288
WINDOW_PATTERN=L
# Training ratio (scaling_params ≈ 700M for d26 with tied-init init):
#   30   = ~21B  tokens  (Chinchilla compute-optimal, fastest)
#   100  = ~70B  tokens  (~23% of data, good baseline, ~25 days)
#   150  = ~105B tokens  (~35% of data, ~35 days)
#   430  = ~300B tokens  (ALL data, ~1 full epoch)
TARGET_RATIO=100
RUN_NAME="attnres-pure-h100-d${DEPTH}"

echo "============================================="
echo " nanochat + AttnRes pure attention (~750M params)"
echo " ${DEPTH} attention layers (pattern: ${WINDOW_PATTERN})"
echo " depth=${DEPTH} | d_model=$((DEPTH * ASPECT_RATIO))"
echo " H100 SXM 80GB HBM3 | batch=${DEVICE_BATCH_SIZE}"
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

# Auto-resume: if a checkpoint exists for this depth, resume from the latest step.
CHECKPOINT_DIR="$HOME/.cache/nanochat/base_checkpoints/d${DEPTH}"
RESUME_ARGS=""
if [ -d "$CHECKPOINT_DIR" ]; then
    LAST_STEP=$(ls "$CHECKPOINT_DIR"/model_*.pt 2>/dev/null \
                | sed -E 's/.*model_0*([0-9]+)\.pt/\1/' \
                | sort -n | tail -1)
    if [ -n "$LAST_STEP" ]; then
        echo "Found checkpoint at step $LAST_STEP -- resuming"
        RESUME_ARGS="--resume-from-step=$LAST_STEP"
    fi
fi

echo "Starting training..."
echo ""

.venv/bin/python -m scripts.base_train \
    $RESUME_ARGS \
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
    --target-param-data-ratio=${TARGET_RATIO} \
    --fp8 \
    --fp8-recipe=tensorwise \
    --use-attn-res \
    --attn-res-block-size=${ATTN_RES_BLOCK_SIZE} \
    --curriculum \
    --curriculum-phase1-ratio=0.6 \
    --curriculum-transition=2000 \
    --run="${RUN_NAME}" \
    --save-every=2000
