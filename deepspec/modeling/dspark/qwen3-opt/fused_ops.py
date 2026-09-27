from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm


@triton.jit
def _rms_norm_forward_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Normalize one row per program without materializing fp32 temporaries."""

    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    input_row = input_ptr + row * n_cols
    output_row = output_ptr + row * n_cols

    values = tl.load(input_row + offsets, mask=mask, other=0.0)
    values_fp32 = values.to(tl.float32)
    variance = tl.sum(values_fp32 * values_fp32, axis=0) / n_cols
    normalized = (values_fp32 * tl.rsqrt(variance + eps)).to(values.dtype)
    weights = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_row + offsets, normalized * weights, mask=mask)


class Qwen3DSparkRMSNorm(Qwen3RMSNorm):
    """Single-kernel CUDA RMSNorm for eager inference."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if (
            self.training
            or torch.is_grad_enabled()
            or not hidden_states.is_cuda
            or not hidden_states.is_contiguous()
        ):
            return super().forward(hidden_states)

        n_cols = hidden_states.shape[-1]
        output = torch.empty_like(hidden_states)
        n_rows = hidden_states.numel() // n_cols
        block_size = triton.next_power_of_2(n_cols)
        num_warps = 4 if block_size < 2048 else 8
        _rms_norm_forward_kernel[(n_rows,)](
            hidden_states,
            self.weight,
            output,
            n_cols=n_cols,
            eps=self.variance_epsilon,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
        return output


def fused_linear(
    module: nn.Module,
    cache_name: str,
    hidden_states: torch.Tensor,
    linears: Sequence[nn.Linear],
) -> torch.Tensor:
    """Evaluate linears sharing an input as one inference GEMM."""

    versions = tuple(
        (linear.weight._version, linear.bias._version if linear.bias is not None else -1)
        for linear in linears
    )
    cache = getattr(module, cache_name, None)
    if (
        cache is None
        or cache[0] != versions
        or cache[1].device != hidden_states.device
        or cache[1].dtype != hidden_states.dtype
    ):
        weight = torch.cat([linear.weight for linear in linears], dim=0)
        biases = [linear.bias for linear in linears]
        bias = None
        if biases[0] is not None:
            bias = torch.cat(biases, dim=0)
        cache = (versions, weight, bias)
        setattr(module, cache_name, cache)
    return F.linear(hidden_states, cache[1], cache[2])


class Qwen3DSparkMLP(Qwen3MLP):
    """Qwen3 MLP with a fused gate/up projection in inference mode."""

    def __init__(self, config):
        super().__init__(config)
        self._gate_up_cache = None

    def train(self, mode: bool = True):
        if mode:
            self._gate_up_cache = None
        return super().train(mode)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.training or torch.is_grad_enabled():
            return super().forward(hidden_states)

        gate_up = fused_linear(
            self,
            "_gate_up_cache",
            hidden_states,
            (self.gate_proj, self.up_proj),
        )
        gate, up = gate_up.split(self.intermediate_size, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)


__all__ = ["Qwen3DSparkMLP", "Qwen3DSparkRMSNorm", "fused_linear"]
