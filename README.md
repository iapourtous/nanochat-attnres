# nanochat-attnres

Research fork of [karpathy/nanochat](https://github.com/karpathy/nanochat) exploring **hybrid conv-attention architectures** with **Attention Residuals** ([arxiv.org/abs/2603.15031](https://arxiv.org/abs/2603.15031)) for bilingual French/English LLM pretraining.

## Architecture

**Hybrid conv-attention model (~978M params).** Not a pure transformer.

The `window_pattern` config string determines layer types at each depth:
- **`S`** = `GatedConv` -- double-gated depthwise causal conv1d (inspired by [Liquid AI / LFM](https://www.liquid.ai/)), ~700x fewer FLOPs than attention
- **`L`** = `CausalSelfAttention` -- full causal attention with RoPE, QK-norm, GQA, Flash Attention 3

Default pattern `SSSL` with depth 26 = **19 conv + 7 attention layers**. Final layer is always attention.

### Attention Residuals (AttnRes)

Replaces standard residual connections (`h_l = h_{l-1} + f(h_{l-1})`) with learned depth-wise softmax attention over block representations. Layers are grouped into blocks; within each block, standard additive residuals; across blocks, learned softmax attention selects which previous blocks to attend to. Block representations are detached (no gradient stored), gradients flow only through the current partial sum.

See `knowledge/summary_attention_residuals.md` for the full paper summary.

### Additional mechanisms

- **Smear gate**: mixes previous token's embedding into current position (cheap bigram-like info) before the transformer stack
- **Backout lambda**: subtracts mid-layer residual before final norm to remove low-level features
- **SwiGLU MLP** with configurable intermediate size
- **GQA** (grouped-query attention) for efficient inference

### Current config (v3 hybrid)

```
depth=26, d_model=1280, n_head=10, n_kv_head=3 (GQA ~3:1)
MLP: SwiGLU, intermediate=4480
RoPE base: 1,000,000
AttnRes block_size=8 (4 layers/block)
Window pattern: SSSL (19 conv + 7 attention)
~978M params
```

## Bilingual data

10 datasets mixing French and English:

| Dataset | Lang | Description |
|---------|------|-------------|
| FineWeb2-HQ | FR | Top 10% quality-filtered French web (~34B tokens) |
| ClimbMix-400B | EN | NVIDIA high-quality English web |
| Wikipedia | FR/EN | Encyclopedic quality |
| PleIAs French-PD-Books | FR | Classic French literature |
| PleIAs French-PD-diverse | FR | Archives & Google Books |
| Europarl | FR/EN | Parallel corpus |
| RedPajama arXiv | EN | Scientific papers |
| MLSUM | FR/EN | News summaries |

Data cached in `~/.cache/nanochat/base_data_bilingual/`, tokenizer (32K vocab) in `~/.cache/nanochat/tokenizer/`.

## Getting started

### Setup

```bash
uv sync --extra gpu          # GPU dependencies (CUDA 12.8)
bash runs/setup_server.sh    # Full setup: deps + data download + tokenizer training + sanity check
```

### Training

```bash
# Single GPU (RTX 5090 32GB)
bash runs/train.sh

# Or manually:
python -m scripts.base_train \
    --depth=26 --aspect-ratio=48 --n-kv-head=3 \
    --intermediate-size=4480 --mlp-type=swiglu --rope-base=1000000 \
    --window-pattern=SSSL --use-attn-res --attn-res-block-size=8 \
    --device-batch-size=5 --total-batch-size=522240 \
    --target-param-data-ratio=80 --run="my-run" --save-every=10000

# Multi-GPU (use -- separator before nanochat flags)
OMP_NUM_THREADS=1 torchrun --nproc_per_node=N -m scripts.base_train -- [flags...]
```

The `--depth` dial controls model size: `d_model = depth * aspect_ratio`, heads auto-computed from `--head-dim`. LR, weight decay, and batch size scale automatically.

### Inference

```bash
python -m scripts.chat_cli     # CLI chat
python -m scripts.chat_web     # Web UI (FastAPI, multi-GPU data parallelism)
```

### Tests

```bash
uv run python -m pytest tests/ -v
uv run python -m pytest tests/test_engine.py -v        # single file
uv run python -m pytest -m "not slow" tests/           # skip slow
```

## Key differences from upstream nanochat

- **Hybrid conv-attention**: GatedConv layers replace most attention layers (pattern-configurable)
- **Attention Residuals**: learned depth-wise softmax attention over block representations
- **Bilingual FR/EN**: 10 datasets vs English-only upstream
- **Smear gate + backout lambda**: additional training signal mechanisms
- **SwiGLU + GQA**: architecture options not in upstream
- **Native FP8 storage** (`nanochat/fp8_native.py`): weights stored as FP8 + scale, ~4x smaller checkpoints
- **Untied embeddings**: wte and lm_head separate (required by AttnRes)

## Optimizer

Combined MuonAdamW: [Muon](https://github.com/KellerJordan/Muon) (orthogonalized momentum) for 2D matrix params, AdamW for embeddings/scalars/1D. LR scaled proportionally to `1/sqrt(d_model)`. Distributed version (`DistMuonAdamW`) with ZeRO-2 style sharding for multi-GPU.

## Research

Experiment log in `dev/LOG.md`. The `autoresearch_program.md` defines the autonomous experiment loop for optimizing AttnRes -- only `nanochat/gpt.py` is modified during research. Key metric: `val_bpb` (bits per byte).

## License

MIT. Based on [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy.
