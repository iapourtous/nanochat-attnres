# Attention Residuals (AttnRes) - Paper Summary

**Paper:** arxiv.org/abs/2603.15031
**Authors:** Kimi Team (Moonshot AI)
**Code:** github.com/MoonshotAI/Attention-Residuals

## Core Idea

Replace fixed residual connections (h_l = h_{l-1} + f(h_{l-1})) with learned depth-wise softmax attention:

h_l = sum_i alpha_{i->l} * v_i

where alpha_{i->l} are softmax attention weights computed from a learned pseudo-query w_l per layer. Keys and values are the layer outputs themselves, with RMSNorm applied to keys.

## Full AttnRes

- Each layer l has a learnable pseudo-query w_l in R^d
- Keys = Values = layer outputs (with RMSNorm on keys)
- Softmax attention over all preceding layer outputs
- O(L^2 d) computation, O(Ld) memory
- Zero-init for queries (uniform attention = standard residuals at start)

## Block AttnRes

- Partition L layers into N blocks of S=L/N layers each
- Intra-block: standard additive residuals to form block representation b_n
- Inter-block: softmax attention over N block representations
- Each sublayer (attn + MLP separately) has its own pseudo-query
- First layer of each block attends over previous blocks only
- Subsequent layers also attend over partial intra-block sum b_n^i
- N~8 recovers most of the benefit
- Memory/communication: O(Nd) instead of O(Ld)

## Key Design Choices (from ablation, Table 5)

| Choice | Loss | Notes |
|--------|------|-------|
| Baseline (PreNorm) | 1.766 | Standard residuals |
| Full AttnRes | 1.737 | Best overall |
| Block AttnRes (S=4) | 1.746 | Good tradeoff |
| w/ input-dependent query | 1.731 | Best but adds d*d params/layer |
| w/ input-independent mixing | 1.749 | Worse than softmax |
| w/ sigmoid (instead of softmax) | 1.741 | Competitive normalization matters |
| w/o RMSNorm on keys | 1.743 (full), 1.750 (block) | RMSNorm important |
| w/ multihead (H=16) | 1.752 | Hurts! Single query better |
| DenseFormer | 1.767 | No gain over baseline |
| mHC | 1.747 | Similar to Block AttnRes |

## Block Size Sweep (16-layer model)

- S=2: 1.746
- S=4: 1.746
- S=8: 1.748
- S=16: 1.753
- S=32: 1.757

Takeaway: S=2 to S=4 is the sweet spot. Larger blocks degrade gracefully.

## Architecture Preference

AttnRes shifts optimal architecture toward **deeper, narrower** models:
- Baseline optimal: d_model/L_b ~ 60
- AttnRes optimal: d_model/L_b ~ 45
- Both prefer H/L_b ~ 0.3 (fewer heads relative to depth)

## Training Details (large-scale)

- Architecture: Kimi Linear 48B (3B activated, MoE)
- 54 layers, 6 layers per block = 9 blocks + embedding = 10 sources
- Optimizer: Muon
- Schedule: WSD (Warmup-Stable-Decay)
- 1T pre-training + 400B mid-training tokens
- Context: 4096 -> 32K extension

## Training Dynamics Benefits

1. **Bounded output magnitudes**: Block boundaries reset accumulation (periodic pattern vs monotonic growth)
2. **Uniform gradient distribution**: Softmax competition distributes gradients more evenly across depth
3. **Mitigates PreNorm dilution**: Each layer's contribution not drowned out by accumulation

## Pseudocode from Paper (Figure 3)

```python
def block_attn_res(blocks, partial_block, proj, norm):
    V = torch.stack(blocks + [partial_block])  # [N+1, B, T, D]
    K = norm(V)
    logits = einsum('d, n b t d -> n b t', proj.weight.squeeze(), K)
    h = einsum('n b t, n b t d -> b t d', logits.softmax(0), V)
    return h

def forward(self, blocks, hidden_states):
    partial_block = hidden_states
    h = block_attn_res(blocks, partial_block, self.attn_res_proj, self.attn_res_norm)
    
    if self.layer_number % (self.block_size // 2) == 0:
        blocks.append(partial_block)
        partial_block = None
    
    attn_out = self.attn(self.attn_norm(h))
    partial_block = partial_block + attn_out if partial_block is not None else attn_out
    
    h = block_attn_res(blocks, partial_block, self.mlp_res_proj, self.mlp_res_norm)
    
    mlp_out = self.mlp(self.mlp_norm(h))
    partial_block = partial_block + mlp_out
    
    return blocks, partial_block
```

## Relevance to nanochat

### Current Implementation Differences

1. **Separate vs shared RMSNorm**: Paper uses per-sublayer RMSNorm module (`self.attn_res_norm`, `self.mlp_res_norm`); nanochat uses shared `F.rms_norm()` inline
2. **Projection vs raw query**: Paper uses `proj.weight.squeeze()` (Linear layer); nanochat uses raw `nn.Parameter` vector -- functionally equivalent
3. **Block boundary logic**: Paper resets `partial_block = None` then adds; nanochat resets to `torch.zeros_like(x)` -- semantically equivalent
4. **Token embedding in blocks**: Paper explicitly includes b_0 = h_1 (token embedding); nanochat includes initial `x` as first block via `partial = x` before loop
5. **Backout lambda**: Not in the paper. nanochat adds mid-layer subtraction on top of AttnRes
6. **resid_lambdas / x0_lambdas**: Not in the paper. nanochat disables them when AttnRes is on
7. **Smear gate**: Not in the paper. nanochat applies previous-token mixing before the transformer
8. **Value embeddings**: Not in the paper. nanochat adds ResFormer-style value embeddings
9. **Activation checkpointing**: nanochat uses grad_checkpoint on attn/MLP but not on block_attn_res itself

### Potential Experiments Informed by Paper

1. The paper shows input-dependent queries give the best loss (1.731 vs 1.737) -- could try in nanochat
2. Block size S=2-4 is optimal -- nanochat uses S=4 (block_size=8 sublayers / 2), which matches
3. Multihead attention over depth hurts -- confirms nanochat's single-query approach is correct
4. RMSNorm on keys is critical -- nanochat correctly includes this
5. Softmax > sigmoid -- nanochat correctly uses softmax
6. Paper does NOT use any backout/resid_lambdas/x0 features -- these may be interfering with AttnRes
