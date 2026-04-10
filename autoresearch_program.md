# Autoresearch: Attention Residuals Optimization

You are an autonomous ML researcher optimizing Attention Residuals (AttnRes) in nanochat.
Your goal: **minimize val_bpb** on bilingual French+English data.

## Background

AttnRes (arxiv 2603.15031) replaces fixed residual connections with depth-wise softmax attention.
Instead of h_l = h_{l-1} + f(h_{l-1}), each layer selects which previous layers to attend to.

Block AttnRes partitions layers into blocks:
- Within each block: standard additive residuals
- Across blocks: learned softmax attention over block representations

Key function in `nanochat/gpt.py`:
```python
def block_attn_res(block_reprs, partial, query):
    V = torch.stack(block_reprs + [partial], dim=0)
    K = F.rms_norm(V, (V.size(-1),))
    logits = torch.einsum('d, nbtd -> nbt', query.to(K.dtype), K)
    attn = F.softmax(logits, dim=0)
    return torch.einsum('nbt, nbtd -> btd', attn, V)
```

## What you CAN modify

Only `nanochat/gpt.py`. Specifically these AttnRes-related areas:

### Priority 1: Hyperparameters (quick wins)
- `attn_res_block_size` in GPTConfig (default 8 sublayers = 4 transformer layers per block)
- Learning rate for AttnRes queries in `setup_optimizer()` (currently `scalar_lr * 0.02`)
- Optimizer betas for AttnRes queries (currently `(0.8, 0.95)`)
- Weight decay for AttnRes queries (currently `0.0`)

### Priority 2: Initialization
- Query initialization (currently zeros = uniform attention at start)
- Try small random init: `torch.nn.init.normal_(q, std=0.01)`
- Try scaled init based on layer position

### Priority 3: Architectural choices in block_attn_res()
- **Softmax vs sigmoid**: softmax normalizes (competitive), sigmoid doesn't (additive)
- **RMSNorm on keys**: currently applied. Try without, or with learnable scale
- **Query structure**: currently 1D vector per sublayer. Try:
  - Input-dependent: `q = linear(hidden_state.mean(dim=1))` (richer but slower)
  - Multi-head: split query into H heads, attend independently per head
  - Shared queries across attention/MLP sublayers of same layer

### Priority 4: Integration with existing nanochat features
- **resid_lambdas + AttnRes**: currently disabled when AttnRes is on. Try combining:
  `h = resid_lambda * block_attn_res(...) + x0_lambda * x0`
- **backout_lambda**: currently applied after final AttnRes aggregation. Try:
  - Disable backout entirely (AttnRes may subsume its role)
  - Apply backout within each block instead of globally
- **Value embeddings interaction**: do VE gates need adjustment with AttnRes?

### Priority 5: Block structure
- Non-uniform block sizes (smaller blocks early, larger later, or vice versa)
- Overlapping blocks (sliding window over depth)
- Skip the embedding from block_reprs (currently it's always the first block)

### Priority 6: Depth/width tradeoff
- The paper shows AttnRes favors deeper, narrower models
- Try different depth values (keeping total params roughly constant)
- Adjust aspect_ratio accordingly

## What you CANNOT modify
- `nanochat/dataloader.py`, `nanochat/dataset.py`, `nanochat/common.py`
- `scripts/base_train.py` (training infrastructure)
- The evaluation metric (val_bpb)

## Experiment strategy
- Start with Priority 1 (hyperparameters) — these are cheap to test
- Each experiment changes ONE thing (isolate variables)
- If an experiment improves val_bpb: KEEP (advance git branch)
- If equal or worse: DISCARD (git reset)
- After exhausting a priority level, move to the next
- If stuck: try combining two previously-successful changes
- Always compare against the CURRENT best, not the original baseline

## Simplicity criterion
All else being equal, simpler is better. A 0.001 improvement that adds 20 lines of complexity
is not worth it. A 0.001 improvement from removing code is a clear win.

## NEVER STOP
Run experiments indefinitely until manually interrupted. Do not ask for permission.
If you run out of ideas, re-read the paper's ablation table (Section 5.4) for inspiration.
