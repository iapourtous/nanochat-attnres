# nanochat-attnres

Research fork of [karpathy/nanochat](https://github.com/karpathy/nanochat) exploring:
1. **Hybrid conv-attention architectures** with gated depthwise convolutions
2. **Attention Residuals** ([arxiv.org/abs/2603.15031](https://arxiv.org/abs/2603.15031)) -- learned depth-wise softmax attention
3. **Latent reasoning preparation** -- groundwork for JEPA-style continuous thinking ([arxiv.org/abs/2512.19171](https://arxiv.org/abs/2512.19171))

Target hardware: single-GPU training on RTX 5090 (32GB), H100 SXM (80GB), or NVIDIA DGX Spark (GB10 Blackwell, 128GB unified).

---

## Architecture

**Hybrid conv-attention model, ~978M params**. Not a pure transformer.

### Layer types (`window_pattern`)

| Char | Layer type | Description |
|------|-----------|-------------|
| `S` | `GatedConv` | Double-gated depthwise causal conv1d, kernel=3, ~700x fewer FLOPs than attention. Inspired by [Liquid AI / LFM](https://www.liquid.ai/) |
| `L` | `CausalSelfAttention` | Full causal attention with RoPE, QK-norm, GQA, Flash Attention 3 (Hopper+) or SDPA fallback |

Default pattern `SSSL` with `depth=32` produces **24 conv + 8 attention layers**. Final layer is always attention.

### Attention Residuals (AttnRes)

Replaces standard residual connections:
```
standard    : h_l = h_{l-1} + f(h_{l-1})              # fixed accumulation
AttnRes     : h_l = softmax(Q_l · K_blocks) · V_blocks # learned depth-wise attention
```

Layers are grouped into blocks of `attn_res_block_size // 2` layers. Within each block, standard additive residuals form a "block representation". Across blocks, each sublayer (attention and MLP) has its own learned pseudo-query that performs softmax attention over completed block representations plus the current partial sum.

**Key mechanics:**
- Block representations are **detached** (no gradient stored) -- saves memory, gradients flow only through `partial`
- Pseudo-queries initialized to **zero** -- uniform attention at start equals standard residuals
- Activation checkpointing on every sublayer

See `knowledge/summary_attention_residuals.md` for full paper summary.

### Additional mechanisms

| Component | Role |
|-----------|------|
| **Smear gate** | Mixes previous token's embedding into current position (cheap bigram-like info) before the transformer stack |
| **Backout lambda** | Subtracts mid-layer residual before final norm to remove low-level features |
| **SwiGLU MLP** | Configurable intermediate size (default 4480) |
| **GQA** | Grouped-query attention with `n_kv_head=3` (GQA 4:1) |
| **Untied embeddings** | `wte` and `lm_head` are separate (required by AttnRes) but tied at **init** for future JEPA Phase 2 |
| **FP8 training** | Drop-in Float8Linear (E4M3 fwd, E5M2 bwd) via `torch._scaled_mm` |
| **FP8 native storage** | `nanochat/fp8_native.py`: weights stored as FP8 + per-tensor scale, ~4x smaller checkpoints |

### Current config (v3 hybrid)

```
depth=32, d_model=1536, n_head=12, n_kv_head=3 (GQA 4:1)
MLP: SwiGLU, intermediate=4480
RoPE base: 1,000,000
AttnRes block_size=8 (4 layers per block, 8 blocks total)
Window pattern: SSSL (24 conv + 8 attention)
Untied embeddings, angularly aligned at init (tied init, untied training)
FP8 training enabled
~978M params (~928M scaling params)
```

---

## JEPA Preparation (Phase 2 groundwork)

This fork lays the groundwork for a future conversion to [JEPA-Reasoner](https://arxiv.org/abs/2512.19171) style latent reasoning, in which the model thinks in a continuous latent space before decoding any tokens.

### Already in place

- **Tied initialization** of `wte` and `lm_head` (angular alignment via `F.normalize` + magnitude preservation). Starts at `cos_sim = 1.0`, drifts down during training but keeps residual correlation -- ideal starting point for JEPA SST.
- **`model/wte_lmhead_cos_sim`** tracked in wandb every 100 steps to observe the divergence.
- **Hybrid conv-attention backbone** -- the Reasoner body is already built. Only the output head needs to be swapped.

### Roadmap (not yet implemented)

The full JEPA pipeline requires three additions:

1. **Phase 2: Self-Supervised Training (SST)** -- replace cross-entropy with scaled cosine distance loss over L2-normalised hidden states, using an EMA teacher for the target. Estimated: ~100 lines in `scripts/jepa_sst.py`.

2. **Talker module** -- a small separate decoder (~200M params) trained on top of a frozen Reasoner to convert latent thoughts back to tokens. Estimated: ~80 lines in `scripts/train_talker.py`.

3. **Latent-first inference** -- `scripts/chat_jepa.py` chains multiple latent thoughts in the Reasoner before invoking the Talker for token production.

### Open research directions

These are intentionally left for exploration:

- **LCM-style concept prediction** -- predict sentence-level embeddings instead of token-level (potential integration with SONAR or self-trained sentence encoder).
- **Recursive AttnRes** -- share weights across layers and let the AttnRes query learn how many loops to apply per token (adaptive depth, inspired by Huginn / Universal Transformers).
- **Mistral embedding init** -- replace trained-from-scratch `wte` with a pretrained multilingual embedding (Mistral 7B v0.3 has matching `vocab_size=32768`).
- **Hard-tied embeddings + AttnRes** -- empirically test whether the assumed gradient conflict actually degrades training.

---

## Bilingual data (curriculum learning)

9 datasets mixing French, English, math, and reasoning content:

| Category | Prefix | Dataset | Notes |
|----------|--------|---------|-------|
| General FR | `fr_` | FineWeb2-HQ French | Top 10% quality-filtered web, ~34B tokens |
| General EN | `en_` | ClimbMix-400B English | NVIDIA high-quality web |
| General FR | `wikifr_` | Wikipedia French | Encyclopedic |
| General EN | `wikien_` | Wikipedia English | Encyclopedic |
| General FR | `booksfr_` | PleIAs French-PD-Books | Classic French literature |
| General FR | `divfr_` | PleIAs French-PD-diverse | Archives & Google Books |
| General | `europarl_` | Europarl FR-EN | Parallel corpus |
| Math | `arxiv_` | RedPajama arXiv | Scientific papers |
| Math (gated) | `nemmath_` | Nemotron-CC-Math v1 4plus | NVIDIA web math, quality 4-5/5 |
| Math | `owm_` | OpenWebMath | StackExchange, MathOverflow, forums |
| Reasoning | `rcore_` | Reasoning-Core SPT | Logic, CSP, graphs, equations, PDDL planning |
| Reasoning | `synloge_`, `synlogh_` | SynLogic easy + hard | 35 logic tasks (Sudoku, cryptarithm, etc.) |
| Docs (skipped) | `docs_` | textbook_quality_programming | Not sampled (too small, would cause memorisation) |

Data cached in `~/.cache/nanochat/base_data_bilingual/`, tokenizer (32K vocab) in `~/.cache/nanochat/tokenizer/`.

### Curriculum phases

Training runs two phases with linear weight transition over `--curriculum-transition` steps.

| Category | Phase 1 (60% of steps) | Phase 2 (40% of steps) |
|----------|------------------------|------------------------|
| general | 65% | 40% |
| math | 28% | 45% |
| reasoning | 7% | 15% |

Within each category, 8 shards are kept open in parallel and a random one is picked at each yield -- strong intra-category mixing to prevent catastrophic forgetting between shards (see `nanochat/dataloader.py`).

---

## Getting started

### Setup

```bash
uv sync --extra gpu          # GPU dependencies (CUDA 12.8)
bash runs/setup_server.sh    # Deps + data download + tokenizer + sanity check
```

On DGX Spark (aarch64, sm_121), the setup script additionally force-installs the PyTorch nightly cu128 aarch64 wheel and warns on swap state.

### Training

```bash
bash runs/train.sh
```

Or manually:
```bash
python -m scripts.base_train \
    --depth=32 --aspect-ratio=48 --n-kv-head=3 \
    --intermediate-size=4480 --mlp-type=swiglu --rope-base=1000000 \
    --window-pattern=SSSL --use-attn-res --attn-res-block-size=8 \
    --device-batch-size=12 --total-batch-size=491520 \
    --target-param-data-ratio=100 \
    --fp8 --fp8-recipe=tensorwise \
    --curriculum --curriculum-phase1-ratio=0.6 --curriculum-transition=2000 \
    --run="my-run" --save-every=2000
```

**Features built in:**
- **Auto-resume**: if a checkpoint exists for the given depth, it resumes from the latest step automatically
- **Checkpoint rotation**: only the latest checkpoint is kept on disk (saves ~5GB per retained checkpoint)
- **Curriculum learning**: 2-phase weighted sampling with linear transition
- **FP8 training**: E4M3/E5M2 via `torch._scaled_mm`, falls back gracefully on non-H100

### Monitoring (wandb)

The training script logs rich diagnostics every 100 steps:

**Standard metrics:**
- `train/loss`, `train/lrm`, `train/dt`, `train/tok_per_sec`, `train/mfu`
- `val/bpb` (bits-per-byte validation loss)
- `core_metric` (DCLM CORE aggregate)

**Learnable scalars** (global modulators):
- `model/smear_lambda` -- previous-token mixing strength
- `model/backout_lambda` -- mid-layer residual subtraction strength
- `model/attn_res_q_final_norm` -- final AttnRes query magnitude
- `model/wte_lmhead_cos_sim` -- angular alignment tracker (starts at 1.0 thanks to tied init, drifts during training)

**Curriculum state:**
- `curriculum/general`, `curriculum/math`, `curriculum/reasoning` -- current phase-blended weights

**AttnRes diagnostics** (opt-in recording):
- `attnres/L<NN>_attn/entropy_norm`, `attnres/L<NN>_mlp/entropy_norm` -- normalized entropy per layer (1.0 = uniform = standard residuals, 0.0 = one-hot selection)
- `attnres/L<NN>_attn/preferred_block`, `attnres/L<NN>_mlp/preferred_block` -- argmax of mean attention
- `attnres/entropy_norm_mean`, `attnres/entropy_norm_min` -- aggregates

### Inference

```bash
python -m scripts.chat_cli     # CLI chat
python -m scripts.chat_web     # Web UI (FastAPI, optional multi-GPU data parallelism)
```

### Tests

```bash
uv run python -m pytest tests/ -v
uv run python -m pytest tests/test_engine.py -v        # single file
uv run python -m pytest -m "not slow" tests/           # skip slow
```

---

## Optimizer

Combined MuonAdamW:
- **[Muon](https://github.com/KellerJordan/Muon)** (orthogonalized momentum via Polar Express iteration) for 2D matrix params
- **AdamW** for embeddings, scalars, 1D params, 3D conv kernels, and AttnRes pseudo-queries
- LR scaled proportionally to `1/sqrt(d_model)` for AdamW groups
- ZeRO-2 style sharding across ranks for multi-GPU (`DistMuonAdamW`)
- Gradient sync handled in the optimizer, no DDP wrapper

---

## Key differences from upstream nanochat

| Area | Upstream | This fork |
|------|----------|-----------|
| Architecture | Pure transformer | Hybrid conv-attention (`SSSL` pattern) |
| Residuals | Fixed additive | Learned depth-wise softmax (AttnRes) |
| Data | English only (ClimbMix) | Bilingual FR/EN + STEM + reasoning (10 sources) |
| Training | Single objective | 2-phase curriculum with weighted sampling |
| Dataloader | Sequential shards | 8-way parallel shard mixing for strong in-category mixing |
| MLP | ReLU² | SwiGLU (configurable) |
| Attention | MHA | GQA (`n_kv_head=3`) |
| Precision | BF16 | BF16 + FP8 (drop-in) + FP8 native storage |
| Initialization | Independent wte/lm_head | **Tied init** (angular alignment, for future JEPA) |
| Checkpointing | Every save interval kept | Single-file rotation + auto-resume |
| Monitoring | Basic | AttnRes entropy + learnable scalars + curriculum weights |

---

## Research context

- **Experiment log**: `dev/LOG.md`
- **Autoresearch loop**: `autoresearch_program.md` -- autonomous optimization agent for AttnRes hyperparameters
- **Key metric**: `val_bpb` (bits per byte) with `core_metric` (DCLM CORE) as aggregate benchmark
- **Paper summary**: `knowledge/summary_attention_residuals.md`

Branches:
- `main` -- production (RTX 5090 / H100 oriented)
- `h100` -- H100 SXM config with FP8 + compile + batch 12
- `dgxSpark` -- aarch64/GB10 config with nightly cu128 and swap guard

---

## License

MIT. Based on [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy.
