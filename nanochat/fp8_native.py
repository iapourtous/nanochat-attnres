"""
Native FP8 Training — weights stored in FP8, not just compute.

Inspired by ECO (arxiv 2601.22101): eliminates FP32 master weights by injecting
quantization error into optimizer momentum (error feedback).

How it works:
1. Model weights are stored as FP8 (E4M3) + per-tensor scale
2. Forward: dequantize to BF16 on the fly → standard compute
3. Backward: gradients computed in BF16 as usual
4. Optimizer step:
   a. Dequantize weight to FP32
   b. Apply standard optimizer update (AdamW/Muon)
   c. Quantize back to FP8
   d. Compute quantization error = (fp32_updated - dequantized_fp8)
   e. Feed error into next step's gradient (error feedback)

Result: model checkpoint is ~4x smaller, directly usable for FP8 inference.
"""

import torch
import torch.nn as nn
from nanochat.common import COMPUTE_DTYPE


FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max  # 448.0


@torch.no_grad()
def quantize_to_fp8(x):
    """Quantize tensor to FP8 E4M3 with dynamic per-tensor scaling."""
    amax = x.float().abs().max().clamp(min=1e-12)
    scale = amax / FP8_MAX
    x_scaled = (x.float() / scale).clamp(-FP8_MAX, FP8_MAX)
    x_fp8 = x_scaled.to(FP8_DTYPE)
    return x_fp8, scale


@torch.no_grad()
def dequantize_from_fp8(x_fp8, scale):
    """Dequantize FP8 tensor to compute dtype."""
    return x_fp8.to(COMPUTE_DTYPE) * scale


class FP8Parameter:
    """
    Wrapper that stores a parameter in FP8 + scale.
    The actual nn.Parameter stays in compute dtype for gradient computation,
    but after each optimizer step we quantize it to FP8 and store the error.
    """
    def __init__(self, param):
        # Quantize the initial weight
        self.fp8_data, self.scale = quantize_to_fp8(param.data)
        # Error feedback buffer (accumulated quantization error)
        self.error = torch.zeros_like(param.data)
        # Reference to the original parameter
        self.param = param

    def dequantize_to_param(self):
        """Write dequantized FP8 weight back into the parameter for forward pass."""
        self.param.data.copy_(dequantize_from_fp8(self.fp8_data, self.scale))

    def quantize_from_param(self):
        """Quantize parameter back to FP8 after optimizer step, with error feedback."""
        # Current full-precision weight after optimizer update
        w_updated = self.param.data.float()
        # Quantize
        self.fp8_data, self.scale = quantize_to_fp8(w_updated)
        # Compute quantization error
        w_reconstructed = self.fp8_data.to(torch.float32) * self.scale
        self.error = w_updated - w_reconstructed

    def state_dict_fp8(self):
        """Return FP8 state for checkpoint saving."""
        return {
            'fp8_data': self.fp8_data,
            'scale': self.scale,
        }

    def load_state_dict_fp8(self, state):
        """Load FP8 state from checkpoint."""
        self.fp8_data = state['fp8_data']
        self.scale = state['scale']
        self.dequantize_to_param()


class FP8NativeTrainer:
    """
    Manages native FP8 training for a model.

    Usage:
        trainer = FP8NativeTrainer(model)
        for step in training:
            trainer.before_forward()      # dequantize FP8 → BF16
            loss = model(x, y)
            loss.backward()
            # inject error feedback into gradients
            trainer.before_optimizer_step()
            optimizer.step()
            trainer.after_optimizer_step() # quantize BF16 → FP8
    """

    def __init__(self, model, min_size=1024):
        """
        Args:
            model: The model to train in native FP8
            min_size: Minimum parameter size to quantize to FP8 (small params stay FP32)
        """
        self.fp8_params = {}  # param_name -> FP8Parameter
        self.model = model

        for name, param in model.named_parameters():
            if param.numel() >= min_size and param.ndim >= 2:
                self.fp8_params[name] = FP8Parameter(param)

        n_fp8 = len(self.fp8_params)
        n_total = sum(1 for _ in model.parameters())
        fp8_numel = sum(p.param.numel() for p in self.fp8_params.values())
        total_numel = sum(p.numel() for p in model.parameters())
        print(f"FP8 Native: {n_fp8}/{n_total} params in FP8 ({fp8_numel/total_numel*100:.1f}% of weights)")
        print(f"  FP8 model size: {fp8_numel / 1e6:.1f}M params × 1 byte = {fp8_numel / 1e9:.2f} GB")
        print(f"  FP32 model size would be: {fp8_numel * 4 / 1e9:.2f} GB")

    def before_forward(self):
        """Dequantize all FP8 weights to BF16 for forward pass."""
        for fp8p in self.fp8_params.values():
            fp8p.dequantize_to_param()

    def before_optimizer_step(self):
        """Inject error feedback into gradients before optimizer step."""
        for fp8p in self.fp8_params.values():
            if fp8p.param.grad is not None:
                # Add previous quantization error to gradient
                # This ensures the optimizer "sees" the accumulated error
                fp8p.param.grad.add_(fp8p.error.to(fp8p.param.grad.dtype))

    def after_optimizer_step(self):
        """Quantize updated weights back to FP8, compute new error."""
        for fp8p in self.fp8_params.values():
            fp8p.quantize_from_param()

    def save_fp8_checkpoint(self, path):
        """Save model with FP8 weights — ~4x smaller than FP32."""
        state = {}
        # Save FP8 weights
        for name, fp8p in self.fp8_params.items():
            state[name + '.fp8_data'] = fp8p.fp8_data.cpu()
            state[name + '.scale'] = fp8p.scale.cpu()
        # Save non-FP8 params as-is (small scalars, etc.)
        for name, param in self.model.named_parameters():
            if name not in self.fp8_params:
                state[name] = param.data.cpu()
        torch.save(state, path)
        size_mb = sum(v.numel() * v.element_size() for v in state.values()) / 1e6
        print(f"Saved FP8 checkpoint: {path} ({size_mb:.0f} MB)")

    def load_fp8_checkpoint(self, path, device='cpu'):
        """Load FP8 checkpoint."""
        state = torch.load(path, map_location=device)
        for name, fp8p in self.fp8_params.items():
            fp8p.fp8_data = state[name + '.fp8_data']
            fp8p.scale = state[name + '.scale']
            fp8p.dequantize_to_param()
        for name, param in self.model.named_parameters():
            if name not in self.fp8_params and name in state:
                param.data.copy_(state[name])
