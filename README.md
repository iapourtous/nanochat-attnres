# nanochat-attnres (a.k.a. nanoInstruct)

Research fork of [karpathy/nanochat](https://github.com/karpathy/nanochat). **Despite the name, this is NOT a chat model** -- it is an **instruct-only model for grounded QA and structured extraction**. The model is trained to answer **only from the provided context** (counterfactual SFT, JEPA latent reasoning, multi-Talker dispatch).

Three research axes:
1. **Hybrid conv-attention architectures** with gated depthwise convolutions (LFM-inspired)
2. **Attention Residuals** ([arxiv.org/abs/2603.15031](https://arxiv.org/abs/2603.15031)) -- learned depth-wise softmax attention
3. **JEPA-Reasoner + multi-Talker** ([arxiv.org/abs/2512.19171](https://arxiv.org/abs/2512.19171)) -- continuous latent thinking with task-specific decoders

Target hardware: single-GPU training on RTX 5090 (32GB), H100 SXM (80GB), or NVIDIA DGX Spark (GB10 Blackwell, 128GB unified).

See [`plan.md`](plan.md) for the full multi-phase training roadmap.

---

## Branches

| Branch | Hardware | Config | Status |
|--------|----------|--------|--------|
| `main` | RTX 5090 (32GB) | d26, ~978M params, no FP8 | maintained |
| `h100` | H100 SXM (80GB), x86_64 | d32, FP8, compile, batch 12 | **active dev** |
| `dgxSpark` | DGX Spark GB10 (128GB unified, aarch64) | d32, FP8, compile, swap guard, nightly cu128 | maintained |

Switch branch with `git checkout <branch>` then `bash runs/setup_server.sh` and `bash runs/train.sh`.

## Project status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Tokenizer with 37 instruct special tokens, tied init | complete |
| 1 | Base pretraining (hybrid SSSL + AttnRes + curriculum + FP8) | **in progress** (~25 days on H100) |
| 2 | SFT format-aware (light, teach the instruct format) | not started |
| 3 | JEPA SST (convert to latent reasoner) | not started |
| 4 | Multi-Talker training (qa / json / triples / classify / summarize) | not started |
| 5 | RL grounding (DPO / GRPO with grounding verifier) | not started |

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

## Instruct format & special tokens (37 total)

Defined in `nanochat/tokenizer.py:SPECIAL_TOKENS`. **No chat tokens** -- the model is instruct-only.

| Category | Tokens | Purpose |
|----------|--------|---------|
| Base | `<\|bos\|>` | Beginning of sequence |
| Context | `<\|context_start\|>`, `<\|context_end\|>` | Wraps the documents/context the model must ground on |
| Input | `<\|input_start\|>`, `<\|input_end\|>` | Wraps the instruction/question |
| Task | `<\|qa\|>`, `<\|extract_json\|>`, `<\|extract_triples\|>`, `<\|classify\|>`, `<\|summarize\|>` | Dispatcher to specialized Talker (Phase 4) |
| Reasoning | `<\|think_start\|>`, `<\|think_end\|>`, `<\|answer_start\|>`, `<\|answer_end\|>`, `<\|no_answer\|>` | Chain-of-thought + grounded answer |
| Structured | `<\|json_start\|>`/`<\|json_end\|>`, `<\|triple_start\|>`/`<\|triple_end\|>`, `<\|class_start\|>`/`<\|class_end\|>`, `<\|summary_start\|>`/`<\|summary_end\|>` | Per-task output wrappers |
| Code | `<\|code_start\|>`, `<\|code_end\|>` | Wrap code snippets in context or output |
| Meta | `<\|citation_start\|>`, `<\|citation_end\|>`, `<\|uncertain\|>` | Citation pointers, uncertainty marker |
| JEPA | `<\|latent\|>` | Placeholder for a latent thought (Phase 3) |
| Reserved | `<\|reserved_0\|>` ... `<\|reserved_7\|>` | 8 future-proof slots |

### Sample format

```
<|bos|>
<|context_start|>
[document(s) the model must ground on]
<|context_end|>
<|qa|>
<|input_start|>[question]<|input_end|>
<|think_start|>
[reasoning STRICTLY from the context]
<|think_end|>
<|answer_start|>[answer]<|answer_end|>      # OR <|no_answer|>
```

### Counterfactual training (Phase 4 highlight)

To teach the model "context > parametric memory", we generate counterfactual samples where the context contradicts known facts:

```
<|context_start|>La tour Eiffel est a Tokyo.<|context_end|>
<|qa|>
<|input_start|>Ou se trouve la tour Eiffel ?<|input_end|>
<|think_start|>Le contexte indique Tokyo.<|think_end|>
<|answer_start|>A Tokyo.<|answer_end|>
```

The model learns that the **context overrides any internalized knowledge** -- a meta-skill that generalizes to any factual question with provided context.

## Multi-Talker architecture (post-Phase 3)

```
┌─────────────────────────────────────────────────────────────────┐
│  REASONER (978M, frozen after JEPA Phase 3)                     │
│  Generates continuous latent thoughts (no tokens yet)           │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼ latents + raw context
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│ Talker_QA    │  │ Talker_JSON  │  │ Talker_Synthesis │
│ ~200M        │  │ ~150M        │  │ ~250M            │
│ Counterfact. │  │ Schema-aware │  │ Summarization    │
└──────────────┘  └──────────────┘  └──────────────────┘
```

Each Talker is small (~100-250M), specialized on one task, and shares the same frozen Reasoner. **You can swap Talkers without re-training the Reasoner.**

## JEPA preparation already in place (in this branch)

- **Tied initialization** of `wte` and `lm_head` (angular alignment via `F.normalize` + magnitude preservation). Starts at `cos_sim = 1.0` -- ideal starting point for JEPA SST.
- **`model/wte_lmhead_cos_sim`** tracked in wandb every 100 steps to observe the divergence.
- **All 37 instruct special tokens in the tokenizer** -- their embeddings are co-trained from the start of Phase 1.
- **Hybrid conv-attention backbone** -- the Reasoner body is already built. Only the output head needs to be swapped (Phase 3).

## Roadmap (not yet implemented)

See [`plan.md`](plan.md) for the full plan. Highlights of what remains:

1. **Phase 2: SFT format-aware** (3-5 days) -- light supervised training to teach the model the instruct format
2. **Phase 3: JEPA SST** (5-10 days) -- convert to latent reasoner via cosine distance + EMA teacher
3. **Phase 4: 5 specialized Talkers** (15-30 days) -- counterfactual QA, JSON, triples, classify, summarize
4. **Phase 5: RL grounding** (optional, 10-20 days) -- DPO/GRPO with grounding verifier

## Open research directions

- **LCM-style concept prediction** -- predict sentence-level embeddings instead of token-level (SONAR or self-trained)
- **Recursive AttnRes** -- share weights across layers, AttnRes query learns how many loops to apply per token (adaptive depth, Huginn / Universal Transformer style)
- **Mistral embedding init** -- replace trained-from-scratch `wte` with Mistral 7B v0.3 (matching `vocab_size=32768`)
- **Hard-tied embeddings + AttnRes** -- empirically test whether the assumed gradient conflict actually degrades training

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

Inference scripts are **not yet implemented**. They will be added once the Talker decoders are trained (Phase 4 in `plan.md`). For now you can use `nanochat/engine.py` as a low-level token-streaming primitive:

```python
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model

model, tokenizer, _ = load_model("base", "d32", step=180000)
engine = Engine(model, tokenizer)

prompt_ids = tokenizer.render_for_completion({
    "task": "qa",
    "context": "La tour Eiffel est a Paris.",
    "input": "Ou est la tour Eiffel ?",
})
results, _ = engine.generate_batch(prompt_ids, max_tokens=64)
print(tokenizer.decode(results[0]))
```

### Tests

```bash
uv run --no-sync python -m pytest tests/ -v
uv run --no-sync python -m pytest tests/test_engine.py -v
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
