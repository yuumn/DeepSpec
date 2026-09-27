from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn
import triton
import triton.language as tl
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm


@triton.jit
def _rms_norm_forward_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_row_stride: tl.constexpr,
    input_group_stride: tl.constexpr,
    group_size: tl.constexpr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Normalize one row per program without materializing fp32 temporaries."""

    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    group = row // group_size
    row_in_group = row - group * group_size
    input_row = (
        input_ptr
        + group * input_group_stride
        + row_in_group * input_row_stride
    )
    output_row = output_ptr + row * n_cols

    values = tl.load(input_row + offsets, mask=mask, other=0.0)
    values_fp32 = values.to(tl.float32)
    variance = tl.sum(values_fp32 * values_fp32, axis=0) / n_cols
    normalized = (values_fp32 * tl.rsqrt(variance + eps)).to(values.dtype)
    weights = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_row + offsets, normalized * weights, mask=mask)


@triton.jit
def _add_rms_norm_forward_kernel(
    residual_ptr,
    update_ptr,
    weight_ptr,
    output_ptr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fuse a BF16 residual addition with the following RMSNorm."""

    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    row_offset = row * n_cols
    residual = tl.load(residual_ptr + row_offset + offsets, mask=mask, other=0.0)
    update = tl.load(update_ptr + row_offset + offsets, mask=mask, other=0.0)
    summed = (residual + update).to(residual.dtype)
    summed_fp32 = summed.to(tl.float32)
    variance = tl.sum(summed_fp32 * summed_fp32, axis=0) / n_cols
    normalized = (summed_fp32 * tl.rsqrt(variance + eps)).to(summed.dtype)
    weights = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + row_offset + offsets, normalized * weights, mask=mask)


@triton.jit
def _concat_k_rms_norm_forward_kernel(
    input0_ptr,
    input1_ptr,
    input2_ptr,
    weight_ptr,
    output_ptr,
    input0_batch_stride: tl.constexpr,
    input0_row_stride: tl.constexpr,
    input1_batch_stride: tl.constexpr,
    input1_row_stride: tl.constexpr,
    input2_batch_stride: tl.constexpr,
    input2_row_stride: tl.constexpr,
    length0: tl.constexpr,
    length1: tl.constexpr,
    length2: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    total_length: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Concatenate K segments, RMS-normalize, and write SDPA's BHSD layout."""

    row = tl.program_id(0)
    head = row % num_heads
    sequence_row = (row // num_heads) % total_length
    batch = row // (num_heads * total_length)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < head_dim

    is_input0 = sequence_row < length0
    is_input1 = (sequence_row >= length0) & (
        sequence_row < length0 + length1
    )
    input0_offset = (
        batch * input0_batch_stride
        + sequence_row * input0_row_stride
        + head * head_dim
    )
    input1_offset = (
        batch * input1_batch_stride
        + (sequence_row - length0) * input1_row_stride
        + head * head_dim
    )
    input2_offset = (
        batch * input2_batch_stride
        + (sequence_row - length0 - length1) * input2_row_stride
        + head * head_dim
    )
    values0 = tl.load(
        input0_ptr + input0_offset + offsets,
        mask=mask & is_input0,
        other=0.0,
    )
    values1 = tl.load(
        input1_ptr + input1_offset + offsets,
        mask=mask & is_input1,
        other=0.0,
    )
    values2 = tl.load(
        input2_ptr + input2_offset + offsets,
        mask=mask & ~(is_input0 | is_input1),
        other=0.0,
    )
    values = values0 + values1 + values2
    values_fp32 = values.to(tl.float32)
    variance = tl.sum(values_fp32 * values_fp32, axis=0) / head_dim
    normalized = (values_fp32 * tl.rsqrt(variance + eps)).to(values.dtype)
    weights = tl.load(weight_ptr + offsets, mask=mask, other=0.0)
    output_offset = (
        (batch * num_heads + head) * total_length + sequence_row
    ) * head_dim
    tl.store(
        output_ptr + output_offset + offsets,
        normalized * weights,
        mask=mask,
    )


class Qwen3MySpecRMSNorm(Qwen3RMSNorm):
    """Single-kernel CUDA RMSNorm for eager inference.

    Training, autograd, CPU, and non-contiguous tensors continue to use the
    Transformers implementation.  Keeping the same module/parameter layout
    also preserves checkpoint compatibility.
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if (
            self.training
            or torch.is_grad_enabled()
            or not hidden_states.is_cuda
            or hidden_states.stride(-1) != 1
        ):
            return super().forward(hidden_states)

        n_cols = hidden_states.shape[-1]
        if hidden_states.ndim == 1:
            group_size = 1
            input_row_stride = n_cols
            input_group_stride = n_cols
        else:
            group_size = hidden_states.shape[-2]
            input_row_stride = hidden_states.stride(-2)
            input_group_stride = group_size * input_row_stride
            if hidden_states.ndim >= 3:
                input_group_stride = hidden_states.stride(-3)
            for dim in range(hidden_states.ndim - 4, -1, -1):
                if hidden_states.stride(dim) != (
                    hidden_states.shape[dim + 1] * hidden_states.stride(dim + 1)
                ):
                    return super().forward(hidden_states)

        output = torch.empty(
            hidden_states.shape,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        n_rows = hidden_states.numel() // n_cols
        block_size = triton.next_power_of_2(n_cols)
        num_warps = 4 if block_size < 2048 else 8
        _rms_norm_forward_kernel[(n_rows,)](
            hidden_states,
            self.weight,
            output,
            input_row_stride=input_row_stride,
            input_group_stride=input_group_stride,
            group_size=group_size,
            n_cols=n_cols,
            eps=self.variance_epsilon,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
        return output

    def forward_add(
        self,
        residual: torch.Tensor,
        update: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.training
            or torch.is_grad_enabled()
            or not residual.is_cuda
            or not residual.is_contiguous()
            or not update.is_contiguous()
            or residual.shape != update.shape
        ):
            return super().forward(residual + update)

        n_cols = residual.shape[-1]
        output = torch.empty_like(residual)
        n_rows = residual.numel() // n_cols
        block_size = triton.next_power_of_2(n_cols)
        num_warps = 4 if block_size < 2048 else 8
        _add_rms_norm_forward_kernel[(n_rows,)](
            residual,
            update,
            self.weight,
            output,
            n_cols=n_cols,
            eps=self.variance_epsilon,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
        return output


def fused_concat_k_rms_norm(
    inputs: Sequence[torch.Tensor],
    norm: Qwen3MySpecRMSNorm,
    head_dim: int,
) -> torch.Tensor:
    """Fuse K concatenation and per-head RMSNorm for inference attention."""

    if len(inputs) not in (2, 3):
        raise ValueError(f"expected two or three K segments, got {len(inputs)}")
    first = inputs[0]
    if (
        norm.training
        or torch.is_grad_enabled()
        or not first.is_cuda
        or any(
            tensor.ndim != 3
            or tensor.shape[0] != first.shape[0]
            or tensor.shape[2] != first.shape[2]
            or tensor.stride(-1) != 1
            for tensor in inputs
        )
        or first.shape[-1] % head_dim != 0
        or norm.weight.numel() != head_dim
    ):
        concatenated = torch.cat(tuple(inputs), dim=1)
        return norm(
            concatenated.view(
                first.shape[0],
                concatenated.shape[1],
                -1,
                head_dim,
            )
        ).transpose(1, 2)

    padded_inputs = (*inputs, inputs[-1])[:3]
    lengths = (*[tensor.shape[1] for tensor in inputs], 0)[:3]
    total_length = sum(lengths)
    num_heads = first.shape[-1] // head_dim
    output = torch.empty(
        first.shape[0],
        num_heads,
        total_length,
        head_dim,
        dtype=first.dtype,
        device=first.device,
    )
    block_size = triton.next_power_of_2(head_dim)
    _concat_k_rms_norm_forward_kernel[
        (first.shape[0] * total_length * num_heads,)
    ](
        *padded_inputs,
        norm.weight,
        output,
        input0_batch_stride=padded_inputs[0].stride(0),
        input0_row_stride=padded_inputs[0].stride(1),
        input1_batch_stride=padded_inputs[1].stride(0),
        input1_row_stride=padded_inputs[1].stride(1),
        input2_batch_stride=padded_inputs[2].stride(0),
        input2_row_stride=padded_inputs[2].stride(1),
        length0=lengths[0],
        length1=lengths[1],
        length2=lengths[2],
        num_heads=num_heads,
        head_dim=head_dim,
        total_length=total_length,
        eps=norm.variance_epsilon,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return output

def fused_linear(
    module: nn.Module,
    cache_name: str,
    hidden_states: torch.Tensor,
    linears: Sequence[nn.Linear],
) -> torch.Tensor:
    """Evaluate linears sharing an input as one GEMM during inference.

    The original ``nn.Linear`` modules remain the source of truth, so checkpoint
    names and the training path are unchanged.  The concatenated weight is a
    non-persistent inference cache and is rebuilt after an in-place parameter
    update or a device/dtype conversion.
    """

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


class Qwen3MySpecMLP(Qwen3MLP):
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


__all__ = [
    "Qwen3MySpecMLP",
    "Qwen3MySpecRMSNorm",
    "fused_concat_k_rms_norm",
    "fused_linear",
]
