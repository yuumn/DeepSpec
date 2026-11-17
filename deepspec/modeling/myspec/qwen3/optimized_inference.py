"""Inference-only DSpark-latentspec RoPE fusion.

Preserves every parameter, attention backend, model layer, token and proposal
operation. Only the two RoPE pointwise chains in each attention layer are
replaced by Triton kernels. Original BF16 intermediate rounding and tensor
strides are retained. No weight packing or input-dependent precomputation.
Call after model.eval(); all subsequent inference must disable autograd.
"""
from types import FunctionType, MethodType
import torch
if __package__:
    from .fused_rope import apply_rotary_pos_emb
else:
    from fused_rope import apply_rotary_pos_emb


def optimize_latentspec(model):
    """Install per-instance fused RoPE; idempotent, does not alter other models."""
    if model.training:
        raise ValueError('Call model.eval() before optimizing inference')
    if getattr(model, '_h800_rope_optimized', False):
        return model
    for layer in [*model.latent_layers, *model.layers]:
        attention = layer.self_attn
        original = attention.forward
        fn = original.__func__
        # Private globals avoid changing the reference implementation globally.
        fused = FunctionType(fn.__code__, dict(fn.__globals__,
            apply_rotary_pos_emb=apply_rotary_pos_emb), fn.__name__,
            fn.__defaults__, fn.__closure__)
        fused.__kwdefaults__ = fn.__kwdefaults__
        attention._h800_original_forward = original
        def guarded(self, *args, _fused=fused, **kwargs):
            if self.training or torch.is_grad_enabled():
                raise RuntimeError('Fused RoPE requires eval and inference_mode/no_grad')
            return _fused(self, *args, **kwargs)
        attention.forward = MethodType(guarded, attention)
    model._h800_rope_optimized = True
    return model


def restore_latentspec(model):
    """Restore the original per-instance attention forwards."""
    for layer in [*model.latent_layers, *model.layers]:
        attention = layer.self_attn
        if hasattr(attention, '_h800_original_forward'):
            attention.forward = attention._h800_original_forward
            del attention._h800_original_forward
    model._h800_rope_optimized = False
    return model
