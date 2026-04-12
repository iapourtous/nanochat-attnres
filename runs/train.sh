#!/bin/bash
# =============================================================================
# Full training: nanochat + AttnRes v3 hybrid conv+attention (~1.5B params)
# Target: 1x NVIDIA DGX Spark (GB10 Blackwell, 128GB unified memory, aarch64)
# Architecture: 24 conv + 8 attention layers (pattern SSSL, depth 32)
# Usage:
#   tmux new -s train
#   bash runs/train.sh
#
# DGX Spark notes:
#   - Unified 128GB memory: GPU OOM = system OOM. Swap must be disabled
#     (done by setup_server.sh) to avoid freezing the machine.
#   - Flash Attention 3 is NOT available on Blackwell; SDPA fallback runs
#     automatically and is actually faster on GB10.
#   - sm_120 kernels (in cu128 wheels) are binary-compatible with sm_121 (GB10).
#     You will see a "sm_121 not supported" warning -- safe to ignore.
#   - torch.compile enabled by default: first compile takes 10-30 min but pays
#     back 30-50% on the long training run. Add --no-compile if it fails.
# =============================================================================
set -e

cd "$(dirname "$0")/.."
# Make sure uv is on PATH
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"
source .venv/bin/activate

# Env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TORCH_CUDA_ARCH_LIST="12.0"   # sm_120 is binary-compatible with sm_121 (GB10)
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Model config — Hybrid Conv+AttnRes (depth 32 ~1.5B params, nudged up from 26/~978M)
DEPTH=32
ASPECT_RATIO=48
# 32 * 48 = 1536 (exact multiple of head_dim=128) → n_head=12, n_kv_head=3 (GQA 4:1)
N_KV_HEAD=3
INTERMEDIATE_SIZE=4480
MLP_TYPE=swiglu
ROPE_BASE=1000000
ATTN_RES_BLOCK_SIZE=8
MAX_SEQ_LEN=2048
# DGX Spark has 128GB unified memory. Cap at 10 to stay well under 80GB peak
# (leaves room for OS + SSH + watchdog, avoids unified-memory OOM freeze).
DEVICE_BATCH_SIZE=10
# 10 * 2048 = 20480, 512000 / 20480 = 25 exact
TOTAL_BATCH_SIZE=512000
WINDOW_PATTERN=SSSL
# Training ratio (recomputed for d32, scaling_params ≈ 930M):
#   30   = ~28B tokens  (~9% of data, Chinchilla-style compute-optimal)
#   100  = ~93B tokens  (~31% of data, same compute budget as d26@150)
#   150  = ~140B tokens (~46% of data)
#   325  = ~302B tokens (ALL data, ~1 full epoch over the whole dataset)
TARGET_RATIO=100
RUN_NAME="attnres-v3-dgxspark-d${DEPTH}"

echo "============================================="
echo " nanochat + AttnRes v3 Hybrid (~1.5B params)"
echo " 24 conv + 8 attention (pattern: ${WINDOW_PATTERN})"
echo " depth=${DEPTH} | d_model=$((DEPTH * ASPECT_RATIO))"
echo " DGX Spark GB10 | batch=${DEVICE_BATCH_SIZE}"
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

# Check swap is off (to prevent OOM freeze of the whole machine)
if [ "$(swapon --show)" != "" ]; then
    echo "WARNING: Swap is enabled. On DGX Spark's unified memory, an OOM can"
    echo "         freeze the machine via a swap-death-spiral. Consider:"
    echo "           sudo swapoff -a"
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
    --save-every=10000
