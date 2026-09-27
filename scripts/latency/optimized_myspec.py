"""Inference-only MySpec operator fusion used by the latency experiment.

This module deliberately lives outside ``deepspec``.  It preserves checkpoint
parameters and replaces only inference execution:

* anchor/latent/mask token embeddings are produced by one embedding lookup;
* projections sharing an input are packed into one GEMM and split afterwards.

Packing is performed once, before timing.  The packed tensors are snapshots of
the model weights and are therefore intended for ``eval()``/inference only.
"""

from __future__ import annotations

from types import MethodType
from typing import Callable

import torch
import torch.nn.functional as F
from transformers import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    eager_attention_forward,
)

from deepspec.modeling.myspec.qwen3.latent_layer import apply_rotary_pos_emb


def _pack_linear_weights(*linears: torch.nn.Linear) -> tuple[torch.Tensor, torch.Tensor | None]:
    weight = torch.cat(
        [linear.weight.detach() for linear in linears], dim=0
    ).contiguous()
    if all(linear.bias is None for linear in linears):
        bias = None
    elif all(linear.bias is not None for linear in linears):
        bias = torch.cat(
            [linear.bias.detach() for linear in linears], dim=0
        ).contiguous()
    else:
        raise ValueError("cannot pack a mixture of biased and bias-free projections")
    return weight, bias


def _attention(
    module,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    implementation = module.config._attn_implementation
    attention_fn: Callable = (
        eager_attention_forward
        if implementation == "eager"
        else ALL_ATTENTION_FUNCTIONS[implementation]
    )
    is_causal = bool(kwargs.get("is_causal", False))
    module.is_causal = is_causal
    kwargs["is_causal"] = is_causal
    return attention_fn(
        module,
        q,
        k,
        v,
        attention_mask,
        dropout=0.0 if not module.training else module.attention_dropout,
        scaling=module.scaling,
        sliding_window=module.sliding_window,
        **kwargs,
    )


def _finish_attention(
    module,
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values,
    cache_position,
    batch_size: int,
    query_length: int,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    key_length = k.shape[1]
    k = k.view(batch_size, key_length, module.num_key_value_heads, module.head_dim)
    v = v.view(batch_size, key_length, module.num_key_value_heads, module.head_dim)
    q = q.view(batch_size, query_length, module.num_attention_heads, module.head_dim)
    q = module.q_norm(q).transpose(1, 2)
    k = module.k_norm(k).transpose(1, 2)
    v = v.transpose(1, 2)
    cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        k, v = past_key_values.update(k, v, module.layer_idx, cache_kwargs)
    output, weights = _attention(module, q, k, v, attention_mask, **kwargs)
    output = output.reshape(
        batch_size, query_length, module.num_attention_heads * module.head_dim
    )
    return module.o_proj(output), weights


def _latent_attention_forward(
    self,
    hidden_states: torch.Tensor,
    target_hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    batch_size, query_length = hidden_states.shape[:-1]
    q_size = self.num_attention_heads * self.head_dim
    kv_size = self.num_key_value_heads * self.head_dim

    q, k_noise, v_noise = F.linear(
        hidden_states, self._fused_q_noise_kv_weight, self._fused_q_noise_kv_bias
    ).split((q_size, kv_size, kv_size), dim=-1)
    k_context, v_context = F.linear(
        target_hidden_states,
        self._fused_context_kv_weight,
        self._fused_context_kv_bias,
    ).split((kv_size, kv_size), dim=-1)
    return _finish_attention(
        self,
        q=q,
        k=torch.cat((k_context, k_noise), dim=1),
        v=torch.cat((v_context, v_noise), dim=1),
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        cache_position=cache_position,
        batch_size=batch_size,
        query_length=query_length,
        **kwargs,
    )


def _latent_attention_forward_packed_kv(
    self,
    hidden_states: torch.Tensor,
    target_hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    batch_size, query_length = hidden_states.shape[:-1]
    q_size = self.num_attention_heads * self.head_dim
    kv_size = self.num_key_value_heads * self.head_dim
    fused_noise = F.linear(
        hidden_states, self._fused_q_noise_kv_weight, self._fused_q_noise_kv_bias
    )
    q = fused_noise[..., :q_size]
    noise_kv = fused_noise[..., q_size:].view(batch_size, query_length, 2, kv_size)
    context_length = target_hidden_states.shape[1]
    context_kv = F.linear(
        target_hidden_states,
        self._fused_context_kv_weight,
        self._fused_context_kv_bias,
    ).view(batch_size, context_length, 2, kv_size)
    packed_kv = torch.cat((context_kv, noise_kv), dim=1)
    return _finish_attention(
        self,
        q=q,
        k=packed_kv[:, :, 0, :],
        v=packed_kv[:, :, 1, :],
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        cache_position=cache_position,
        batch_size=batch_size,
        query_length=query_length,
        **kwargs,
    )


def _mask_attention_forward(
    self,
    hidden_states: torch.Tensor,
    target_hidden_states: torch.Tensor,
    latent_hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    batch_size, query_length = hidden_states.shape[:-1]
    q_size = self.num_attention_heads * self.head_dim
    kv_size = self.num_key_value_heads * self.head_dim

    q, k_mask, v_mask = F.linear(
        hidden_states, self._fused_q_mask_kv_weight, self._fused_q_mask_kv_bias
    ).split((q_size, kv_size, kv_size), dim=-1)
    k_context, v_context = F.linear(
        target_hidden_states,
        self._fused_context_kv_weight,
        self._fused_context_kv_bias,
    ).split((kv_size, kv_size), dim=-1)
    k_latent, v_latent = F.linear(
        latent_hidden_states,
        self._fused_latent_kv_weight,
        self._fused_latent_kv_bias,
    ).split((kv_size, kv_size), dim=-1)
    return _finish_attention(
        self,
        q=q,
        k=torch.cat((k_context, k_latent, k_mask), dim=1),
        v=torch.cat((v_context, v_latent, v_mask), dim=1),
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        cache_position=cache_position,
        batch_size=batch_size,
        query_length=query_length,
        **kwargs,
    )


def _mask_attention_forward_packed_kv(
    self,
    hidden_states: torch.Tensor,
    target_hidden_states: torch.Tensor,
    latent_hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    batch_size, query_length = hidden_states.shape[:-1]
    q_size = self.num_attention_heads * self.head_dim
    kv_size = self.num_key_value_heads * self.head_dim
    fused_mask = F.linear(
        hidden_states, self._fused_q_mask_kv_weight, self._fused_q_mask_kv_bias
    )
    q = fused_mask[..., :q_size]
    mask_kv = fused_mask[..., q_size:].view(batch_size, query_length, 2, kv_size)
    context_length = target_hidden_states.shape[1]
    context_kv = F.linear(
        target_hidden_states,
        self._fused_context_kv_weight,
        self._fused_context_kv_bias,
    ).view(batch_size, context_length, 2, kv_size)
    latent_length = latent_hidden_states.shape[1]
    latent_kv = F.linear(
        latent_hidden_states,
        self._fused_latent_kv_weight,
        self._fused_latent_kv_bias,
    ).view(batch_size, latent_length, 2, kv_size)
    packed_kv = torch.cat((context_kv, latent_kv, mask_kv), dim=1)
    return _finish_attention(
        self,
        q=q,
        k=packed_kv[:, :, 0, :],
        v=packed_kv[:, :, 1, :],
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        cache_position=cache_position,
        batch_size=batch_size,
        query_length=query_length,
        **kwargs,
    )


def optimize_myspec_for_inference(
    model: torch.nn.Module, *, packed_kv_cat: bool = False
) -> torch.nn.Module:
    """Pack projection weights and install fused attention forwards in-place."""
    if model.training:
        raise ValueError("optimization is inference-only; call model.eval() first")
    if getattr(model, "_latency_fusion_enabled", False):
        return model

    for layer in model.latent_layers:
        attention = layer.self_attn
        (
            attention._fused_q_noise_kv_weight,
            attention._fused_q_noise_kv_bias,
        ) = _pack_linear_weights(
            attention.q_proj, attention.k_proj_noise, attention.v_proj_noise
        )
        (
            attention._fused_context_kv_weight,
            attention._fused_context_kv_bias,
        ) = _pack_linear_weights(attention.k_proj, attention.v_proj)
        attention.forward = MethodType(
            _latent_attention_forward_packed_kv
            if packed_kv_cat
            else _latent_attention_forward,
            attention,
        )

    for layer in model.layers:
        attention = layer.self_attn
        (
            attention._fused_q_mask_kv_weight,
            attention._fused_q_mask_kv_bias,
        ) = _pack_linear_weights(
            attention.q_proj, attention.k_proj_mask, attention.v_proj_mask
        )
        (
            attention._fused_context_kv_weight,
            attention._fused_context_kv_bias,
        ) = _pack_linear_weights(attention.k_proj, attention.v_proj)
        (
            attention._fused_latent_kv_weight,
            attention._fused_latent_kv_bias,
        ) = _pack_linear_weights(attention.k_proj_latent, attention.v_proj_latent)
        attention.forward = MethodType(
            _mask_attention_forward_packed_kv
            if packed_kv_cat
            else _mask_attention_forward,
            attention,
        )

    model._latency_fusion_enabled = True
    model._packed_kv_cat_enabled = packed_kv_cat
    return model


def embed_anchor_latent_mask_once(
    model: torch.nn.Module,
    *,
    anchor_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Produce latent and mask embeddings with one lookup and one anchor read."""
    batch_size = anchor_ids.shape[0]
    latent_tail = torch.full(
        (batch_size, int(model.num_latent_tokens) - 1),
        int(model.latent_token_id),
        device=anchor_ids.device,
        dtype=anchor_ids.dtype,
    )
    mask_tail = torch.full(
        (batch_size, int(model.block_size) - 1),
        int(model.mask_token_id),
        device=anchor_ids.device,
        dtype=anchor_ids.dtype,
    )
    all_ids = torch.cat((anchor_ids, latent_tail, mask_tail), dim=1)
    all_embeddings = model.embed_tokens(all_ids)
    latent_end = int(model.num_latent_tokens)
    anchor_embedding = all_embeddings[:, :1]
    latent_embedding = all_embeddings[:, :latent_end]
    mask_embedding = torch.cat((anchor_embedding, all_embeddings[:, latent_end:]), dim=1)
    return latent_embedding, mask_embedding


def forward_optimized_myspec_draft_block(
    model: torch.nn.Module,
    *,
    anchor_ids: torch.Tensor,
    position_ids: torch.Tensor,
    past_key_values_draft: DynamicCache,
    target_hidden_states: torch.Tensor,
    start: int,
    block_size: int,
) -> torch.Tensor:
    """Equivalent of ``forward_myspec_draft_block`` using fused embeddings."""
    if block_size != int(model.block_size):
        raise ValueError("optimized path requires the model's configured block size")
    cache_length = past_key_values_draft.get_seq_length()
    if cache_length > start:
        raise ValueError("draft cache cannot be longer than the accepted prefix")

    latent_embedding, mask_embedding = embed_anchor_latent_mask_once(
        model, anchor_ids=anchor_ids
    )
    cached_positions = position_ids[:, cache_length:start]
    latent_positions = position_ids[:, start : start + model.num_latent_tokens]
    mask_positions = position_ids[:, start : start + block_size]
    latent_position_ids = torch.cat((cached_positions, latent_positions), dim=1)
    full_position_ids = torch.cat(
        (cached_positions, latent_positions, mask_positions), dim=1
    )
    hidden = model._forward_backbone(
        target_hidden_states=target_hidden_states,
        latent_noise_embedding=latent_embedding,
        latent_position_ids=latent_position_ids,
        noise_embedding=mask_embedding,
        position_ids=full_position_ids,
        attention_mask=None,
        past_key_values=past_key_values_draft,
        use_cache=True,
        is_causal=False,
    )
    past_key_values_draft.crop(start)
    return hidden
