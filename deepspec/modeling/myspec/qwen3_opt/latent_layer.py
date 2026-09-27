from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn

from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    eager_attention_forward,
    rotate_half,
)
from typing_extensions import Tuple, Unpack

from .fused_ops import (
    Qwen3MySpecMLP,
    Qwen3MySpecRMSNorm,
    fused_concat_k_rms_norm,
    fused_linear,
)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Qwen3MySpecAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (
            self.num_attention_heads // self.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self._cuda_graph_pack_kv = False
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj_noise = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj_noise = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3MySpecRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3MySpecRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        context_key_value_states: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden_states.shape[1]
        use_inference_fusion = not self.training and not torch.is_grad_enabled()
        pack_kv = self._cuda_graph_pack_kv and ctx_len < 4096
        if use_inference_fusion and (ctx_len <= 4096 or pack_kv):
            hidden_qkv = fused_linear(
                self,
                "_hidden_qkv_cache",
                hidden_states,
                (self.q_proj, self.k_proj_noise, self.v_proj_noise),
            )
            q_size = self.num_attention_heads * self.head_dim
            kv_size = self.num_key_value_heads * self.head_dim
            q = hidden_qkv[..., :q_size]
            noise_kv = hidden_qkv[..., q_size:].view(
                bsz, q_len, 2, kv_size
            )
            k_noise, v_noise = noise_kv.unbind(dim=2)
        else:
            noise_kv = None
            q = self.q_proj(hidden_states)
            k_noise = self.k_proj_noise(hidden_states)
            v_noise = self.v_proj_noise(hidden_states)

        # Combining the context projections avoids a launch for short
        # prompts.  On H20, two narrower GEMMs are faster once the context is
        # large enough to saturate the device.
        if isinstance(context_key_value_states, torch.Tensor):
            context_kv = context_key_value_states
            k_ctx, v_ctx = context_kv.unbind(dim=2)
        elif context_key_value_states is not None:
            context_kv = None
            k_ctx, v_ctx = context_key_value_states
        elif use_inference_fusion and ctx_len <= 4096:
            context_kv_flat = fused_linear(
                self,
                "_context_kv_cache",
                target_hidden_states,
                (self.k_proj, self.v_proj),
            )
            context_kv = context_kv_flat.view(bsz, ctx_len, 2, -1)
            k_ctx, v_ctx = context_kv.unbind(dim=2)
        else:
            context_kv = None
            k_ctx = self.k_proj(target_hidden_states)
            v_ctx = self.v_proj(target_hidden_states)

        q = q.view(
            bsz, q_len, self.num_attention_heads, self.head_dim
        )
        q = self.q_norm(q).transpose(1, 2)
        fused_long_k = self._cuda_graph_pack_kv and ctx_len >= 4096
        if fused_long_k:
            k = fused_concat_k_rms_norm(
                (k_ctx, k_noise),
                self.k_norm,
                self.head_dim,
            )
            v = torch.cat([v_ctx, v_noise], dim=1).view(
                bsz,
                ctx_len + q_len,
                self.num_key_value_heads,
                self.head_dim,
            ).transpose(1, 2)
        elif pack_kv and context_kv is not None and noise_kv is not None:
            packed_kv = torch.cat(
                (
                    context_kv.permute(2, 0, 1, 3),
                    noise_kv.permute(2, 0, 1, 3),
                ),
                dim=2,
            )
            k, v = packed_kv.unbind(dim=0)
        else:
            k = torch.cat([k_ctx, k_noise], dim=1)
            v = torch.cat([v_ctx, v_noise], dim=1)
        if not fused_long_k:
            k = k.view(bsz, ctx_len + q_len, self.num_key_value_heads, self.head_dim)
            v = v.view(bsz, ctx_len + q_len, self.num_key_value_heads, self.head_dim)
            k = self.k_norm(k).transpose(1, 2)
            v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        # if (
        #     self.config._attn_implementation == "flex_attention"
        #     and self.num_key_value_groups > 1
        # ):
        #     kv_seq_len = k.shape[-2]
        #     k = k.repeat_interleave(self.num_key_value_groups, dim=1)
        #     v = v.repeat_interleave(self.num_key_value_groups, dim=1)
        #     k = k.reshape(bsz, self.num_attention_heads, kv_seq_len, self.head_dim)
        #     v = v.reshape(bsz, self.num_attention_heads, kv_seq_len, self.head_dim)
        # attn_fn: Callable = eager_attention_forward
        # if self.config._attn_implementation != "eager":
        #     attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_impl = self.config._attn_implementation
        attn_is_causal = bool(kwargs.get("is_causal", False))
        if attn_impl == "sdpa" and not self.training and attention_mask is None:
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=0.0,
                scale=self.scaling,
                is_causal=attn_is_causal,
                enable_gqa=self.num_key_value_groups > 1,
            ).transpose(1, 2).contiguous()
            attn_weights = None
        else:
            attn_fn: Callable = (
                eager_attention_forward
                if attn_impl == "eager"
                else ALL_ATTENTION_FUNCTIONS[attn_impl]
            )
            # The SDPA path may consult module.is_causal when dispatching kernels,
            # so keep the per-call value mirrored on the module before invoking it.
            self.is_causal = attn_is_causal
            kwargs["is_causal"] = attn_is_causal
            attn_output, attn_weights = attn_fn(
                self,
                q,
                k,
                v,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )
        attn_output = attn_output.reshape(
            bsz, q_len, self.num_attention_heads * self.head_dim
        )
        return self.o_proj(attn_output), attn_weights


class Qwen3MySpecLatentDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3MySpecAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MySpecMLP(config)
        self.input_layernorm = Qwen3MySpecRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3MySpecRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        target_hidden_states: Optional[torch.Tensor] = None,
        context_key_value_states: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        output_norm: Optional[Qwen3MySpecRMSNorm] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden_states=target_hidden_states,
            context_key_value_states=context_key_value_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if output_norm is not None:
            return output_norm.forward_add(residual, hidden_states)
        return residual + hidden_states
