"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    intermediate_size: int = -1 # MLP intermediate dim (-1 = auto: 4*n_embd for ReLU², 8*n_embd//3 for SwiGLU)
    mlp_type: str = "relu2" # "relu2" (ReLU squared) or "swiglu" (SiLU gated linear unit)
    rope_base: float = 100000.0 # RoPE base frequency (100K standard, 1M for better long-context)
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"
    # Attention Residuals (AttnRes): replace fixed residual accumulation with learned
    # depth-wise softmax attention over block representations. See arxiv.org/abs/2603.15031
    use_attn_res: bool = False
    attn_res_block_size: int = 8 # sublayer count per block (each transformer layer = 2 sublayers)


def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok

def block_attn_res(block_reprs, partial, query):
    """
    Depth-wise softmax attention over block representations (Attention Residuals).
    Block representations are detached (no gradient) — gradients flow through partial only.

    Args:
        block_reprs: list of (B, T, D) tensors — completed block sums (detached)
        partial: (B, T, D) — current intra-block partial sum (has gradient)
        query: (D,) — learned pseudo-query for this depth position
    Returns:
        (B, T, D) — attention-weighted aggregation
    """
    if not block_reprs:
        return partial
    V = torch.stack(block_reprs + [partial], dim=0)  # (N+1, B, T, D)
    K = F.rms_norm(V, (V.size(-1),))
    logits = torch.einsum('d, nbtd -> nbt', query.to(K.dtype), K)
    attn = F.softmax(logits, dim=0)
    return torch.einsum('nbt, nbtd -> btd', attn, V)


class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm
        q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        k = k * 1.2

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.n_embd
        if config.intermediate_size > 0:
            intermediate = config.intermediate_size
        else:
            intermediate = 4 * d if config.mlp_type == "relu2" else (8 * d // 3 + 127) // 128 * 128
        self.mlp_type = config.mlp_type
        if config.mlp_type == "swiglu":
            self.gate = Linear(d, intermediate, bias=False)
            self.up = Linear(d, intermediate, bias=False)
            self.down = Linear(intermediate, d, bias=False)
        else:
            self.c_fc = Linear(d, intermediate, bias=False)
            self.c_proj = Linear(intermediate, d, bias=False)

    def forward(self, x):
        if self.mlp_type == "swiglu":
            return self.down(F.silu(self.gate(x)) * self.up(x))
        else:
            x = self.c_fc(x)
            x = F.relu(x).square()
            x = self.c_proj(x)
            return x


class GatedConv(nn.Module):
    """Gated depthwise short convolution (inspired by LFM 2.5 / Liquid AI).
    Double-gated: input gate B and output gate C around a causal depthwise conv.
    Replaces attention for local context mixing at ~700x less FLOPs per layer."""
    def __init__(self, config):
        super().__init__()
        d = config.n_embd
        self.gate_b = Linear(d, d, bias=False)
        self.gate_c = Linear(d, d, bias=False)
        self.conv = nn.Conv1d(d, d, kernel_size=3, padding=0, groups=d, bias=False)
        self.out_proj = Linear(d, d, bias=False)

    def forward(self, x):
        b_gate = torch.sigmoid(self.gate_b(x))
        c_gate = torch.sigmoid(self.gate_c(x))
        y = b_gate * x
        y = y.transpose(1, 2)
        y = F.pad(y, (2, 0))
        y = F.conv1d(y, self.conv.weight.to(dtype=y.dtype), groups=y.size(1))
        y = y.transpose(1, 2)
        return self.out_proj(c_gate * y)


class Block(nn.Module):
    def __init__(self, config, layer_idx, block_type='attention'):
        super().__init__()
        self.block_type = block_type
        if block_type == 'conv':
            self.conv = GatedConv(config)
        else:
            self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
        # AttnRes: per-sublayer pseudo-queries
        if config.use_attn_res:
            self.attn_res_q_attn = nn.Parameter(torch.zeros(config.n_embd))
            self.attn_res_q_mlp = nn.Parameter(torch.zeros(config.n_embd))

    def forward(self, x, cos_sin=None, window_size=None, kv_cache=None):
        if self.block_type == 'conv':
            x = x + self.conv(norm(x))
        else:
            x = x + self.attn(norm(x), cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (seq_len, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        # Determine block types from window_pattern: S=conv, L=attention
        pattern = config.window_pattern.upper()
        self.block_types = []
        for i in range(config.n_layer):
            char = pattern[i % len(pattern)]
            self.block_types.append('conv' if char == 'S' else 'attention')
        self.block_types[-1] = 'attention'  # final layer always attention
        n_attn = sum(1 for t in self.block_types if t == 'attention')
        n_conv = sum(1 for t in self.block_types if t == 'conv')
        print0(f"Hybrid architecture: {n_attn} attention + {n_conv} conv layers (pattern: {pattern})")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx, self.block_types[layer_idx]) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # AttnRes: final aggregation query (only when enabled)
        if config.use_attn_res:
            self.attn_res_q_final = nn.Parameter(torch.zeros(config.n_embd))
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout: subtract cached mid-layer residual before final norm to remove low-level features
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            if block.block_type == 'conv':
                # Conv block init
                torch.nn.init.uniform_(block.conv.gate_b.weight, -s, s)
                torch.nn.init.uniform_(block.conv.gate_c.weight, -s, s)
                torch.nn.init.ones_(block.conv.conv.weight)  # identity-like init for depthwise
                block.conv.conv.weight.data /= 3.0  # average over kernel
                torch.nn.init.zeros_(block.conv.out_proj.weight)
            else:
                # Attention block init
                torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
                torch.nn.init.zeros_(block.attn.c_proj.weight)
            # MLP init (same for both block types)
            if self.config.mlp_type == "swiglu":
                torch.nn.init.uniform_(block.mlp.gate.weight, -s * 0.4, s * 0.4)
                torch.nn.init.uniform_(block.mlp.up.weight, -s * 0.4, s * 0.4)
                torch.nn.init.zeros_(block.mlp.down.weight)
            else:
                torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)
                torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        # Per-layer resid init: stronger residual at early layers, weaker at deep layers
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        # Decaying x0 init: earlier layers get more input embedding blending
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        # AttnRes: zero-init all pseudo-queries (uniform attention = standard residuals at start)
        if self.config.use_attn_res:
            for block in self.transformer.h:
                torch.nn.init.zeros_(block.attn_res_q_attn)
                torch.nn.init.zeros_(block.attn_res_q_mlp)
            torch.nn.init.zeros_(self.attn_res_q_final)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=None, device=None):
        if base is None:
            base = self.config.rope_base
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (quarter context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 512)
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        nparams_exclude = (self.transformer.wte.weight.numel() +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                          self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel()
        attn_res = self.attn_res_q_final.numel() if self.config.use_attn_res else 0
        total = wte + lm_head + transformer_matrices + scalars + attn_res
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'attn_res': attn_res,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        # Extract non-Muon params: 1D (AttnRes pseudo-queries) and 3D (Conv1d depthwise kernels)
        # Muon only works on 2D matrices
        attn_res_params = []
        conv_params = [p for p in matrix_params if p.ndim == 3]  # Conv1d weights (d, 1, k)
        matrix_params = [p for p in matrix_params if p.ndim == 2]  # 2D only for Muon
        if self.config.use_attn_res:
            attn_res_params = [p for p in self.transformer.h.parameters() if p.ndim == 1]
            attn_res_params.append(self.attn_res_q_final)
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(resid_params) + len(x0_params) + len(smear_params) + len(attn_res_params) + len(conv_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # AttnRes pseudo-queries: AdamW with moderate LR
        if attn_res_params:
            param_groups.append(dict(kind='adamw', params=attn_res_params, lr=scalar_lr * 0.02, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0))
        # Conv1d depthwise kernels: AdamW (3D, not compatible with Muon)
        if conv_params:
            param_groups.append(dict(kind='adamw', params=conv_params, lr=matrix_lr * dmodel_lr_scale, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0))
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Embed the tokens
        x = self.transformer.wte(idx) # embed current token
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = norm(x)

        # Smear: mix previous token's embedding into current position (cheap bigram info)
        if kv_cache is None:
            # Training / naive generate: full sequence available, use fast slice
            if T > 1:
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            # KV cache inference: read prev embedding from cache, store current for next step
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                # Prefill: apply smear to positions 1+, same as training
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                # Decode: single token, use cached prev embedding
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        # Forward the trunk of the Transformer
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2  # cache at halfway point
        x_backout = None

        if self.config.use_attn_res:
            # --- Attention Residuals path ---
            # Block representations are detached (saves memory, gradients flow via partial).
            # Activation checkpointing on attention + MLP sublayers.
            assert self.config.attn_res_block_size >= 2, f"attn_res_block_size must be >= 2, got {self.config.attn_res_block_size}"
            layers_per_block = self.config.attn_res_block_size // 2
            block_reprs = []  # detached block representations (no gradient stored)
            partial = x

            for i, layer in enumerate(self.transformer.h):
                # Pre-attention/conv: depth-wise attention over blocks + partial
                h = block_attn_res(block_reprs, partial, layer.attn_res_q_attn)

                # Block boundary: store detached partial as completed block
                if i % layers_per_block == 0:
                    block_reprs.append(partial.detach())
                    partial = torch.zeros_like(x)

                # Attention or Conv sublayer (with activation checkpointing)
                if layer.block_type == 'conv':
                    sub_out = grad_checkpoint(layer.conv, norm(h), use_reentrant=False)
                else:
                    sub_out = grad_checkpoint(layer.attn, norm(h), cos_sin, self.window_sizes[i], None, use_reentrant=False)
                partial = partial + sub_out

                # Pre-MLP: depth-wise attention over blocks + partial
                h = block_attn_res(block_reprs, partial, layer.attn_res_q_mlp)

                # MLP sublayer (with activation checkpointing)
                mlp_out = grad_checkpoint(layer.mlp, norm(h), use_reentrant=False)
                partial = partial + mlp_out

                if i == backout_layer:
                    x_backout = block_attn_res(block_reprs, partial, self.attn_res_q_final)

            # Final aggregation
            x = block_attn_res(block_reprs, partial, self.attn_res_q_final)
        else:
            # --- Standard residual path (original nanochat) ---
            x0 = x  # save initial normalized embedding for x0 residual
            for i, block in enumerate(self.transformer.h):
                x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
                if block.block_type == 'conv':
                    x = block(x)
                else:
                    x = block(x, cos_sin, self.window_sizes[i], kv_cache)
                if i == backout_layer:
                    x_backout = x

        # Subtract mid-layer residual to remove low-level features before logit projection
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference: just return the logits directly
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
