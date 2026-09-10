from typing import Callable, Optional

import torch
from torch import nn

from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)
from typing_extensions import Tuple, Unpack

from deepspec.modeling.myspec.common import (
    AcceptRatePredictor,
    MySpecForwardOutput,
    build_eval_mask,
    create_myspec_latent_attention_mask,
    create_myspec_mask_attention_mask,
    create_noise_embed,
    create_position_ids,
    log_sampler_stats,
    sample_anchor_positions,
    create_latent_position_ids,
    create_latent_noise_embed,
)
from deepspec.modeling.myspec.qwen3.mask_layer import Qwen3MySpecMaskDecoderLayer
from deepspec.modeling.myspec.qwen3.latent_layer import Qwen3MySpecLatentDecoderLayer
from deepspec.modeling.myspec.markov_head import build_markov_head
from deepspec.utils.sampling import sample_tokens


class Qwen3MySpecModel(Qwen3PreTrainedModel):
    _no_split_modules = ["Qwen3MySpecLatentDecoderLayer", "Qwen3MySpecMaskDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        required_fields = (
            "target_layer_ids",
            "mask_token_id",
            "num_anchors",
            "enable_confidence_head",
            "markov_rank",
        )
        for field in required_fields:
            assert hasattr(config, field), f"config.{field} must be provided."
        if int(config.markov_rank) > 0:
            assert hasattr(config, "markov_head_type"), (
                "config.markov_head_type must be provided when markov_rank > 0."
            )
        if bool(config.enable_confidence_head):
            assert hasattr(config, "confidence_head_with_markov"), (
                "config.confidence_head_with_markov must be provided when "
                "enable_confidence_head is true."
            )
        self.target_layer_ids = config.target_layer_ids

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=getattr(config, "pad_token_id", None),
        )

        # Latent Cot
        self.num_latent_layers = int(config.num_latent_layers)
        self.num_latent_tokens = int(config.num_latent_tokens)
        self.latent_token_id = int(config.latent_token_id)

        self.latent_layers = nn.ModuleList(
            [
                Qwen3MySpecLatentDecoderLayer(config, layer_idx)
                for layer_idx in range(self.num_latent_layers)
            ]
        )

        self.layers = nn.ModuleList(
            [
                Qwen3MySpecMaskDecoderLayer(config, layer_idx + self.num_latent_layers, self.num_latent_layers)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.latent_hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.block_size = int(config.block_size)
        self.mask_token_id = config.mask_token_id
        self.num_anchors = int(config.num_anchors)

        # Markov head.
        self.markov_head = build_markov_head(config)

        # Confidence head.
        self.enable_confidence_head = bool(config.enable_confidence_head)
        self.confidence_head_with_markov = False
        if self.enable_confidence_head:
            self.confidence_head_with_markov = bool(config.confidence_head_with_markov)
        if self.enable_confidence_head and self.confidence_head_with_markov:
            assert self.markov_head is not None

        self.confidence_head = None
        if self.enable_confidence_head:
            input_dim = int(config.hidden_size)
            if self.confidence_head_with_markov:
                input_dim += config.markov_rank
            self.confidence_head = AcceptRatePredictor(input_dim=input_dim)
        self.post_init()

    def initialize_embeddings_and_head(
        self,
        *,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        freeze: bool = True,
    ):
        assert self.embed_tokens.weight.shape == embed_tokens.weight.shape
        assert self.lm_head.weight.shape == lm_head.weight.shape
        with torch.no_grad():
            self.embed_tokens.weight.copy_(embed_tokens.weight.detach())
            self.lm_head.weight.copy_(lm_head.weight.detach())
        if freeze:
            self.set_embedding_head_trainable(False)

    def set_embedding_head_trainable(self, trainable: bool):
        self.embed_tokens.requires_grad_(trainable)
        self.lm_head.requires_grad_(trainable)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def predict_confidence_step(
        self,
        hidden_states: torch.Tensor,
        prev_token_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if self.confidence_head is None:
            return None
        if self.confidence_head_with_markov:
            assert self.markov_head is not None
            assert prev_token_ids is not None
            prev_embeddings = self.markov_head.get_prev_embeddings(prev_token_ids).to(
                dtype=hidden_states.dtype
            )
            features = torch.cat([hidden_states, prev_embeddings], dim=-1)
            return self.confidence_head(features).float()
        return self.confidence_head(hidden_states).float()

    def sample_draft_tokens(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_token_ids: torch.Tensor,
        temperature: float = 0.0,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty_tokens = torch.empty(
                batch_size,
                0,
                dtype=torch.long,
                device=base_logits.device,
            )
            return empty_tokens, base_logits
        if self.markov_head is None:
            return sample_tokens(base_logits, temperature), base_logits
        return self.markov_head.sample_block_tokens(
            base_logits,
            first_prev_token_ids=first_prev_token_ids,
            hidden_states=hidden_states,
            temperature=temperature,
        )

    def sample_draft_token_step(
        self,
        base_logits: torch.Tensor,
        *,
        prev_token_ids: torch.Tensor,
        temperature: float = 0.0,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert base_logits.ndim == 2, (
            "sample_draft_token_step expects base_logits shaped [batch, vocab], "
            f"got {tuple(base_logits.shape)}."
        )
        if self.markov_head is None:
            step_logits = base_logits
        else:
            step_logits = self.markov_head.apply_step_logits(
                base_logits,
                token_ids=prev_token_ids,
                hidden_states=hidden_states,
            )
        sampled_token_ids = sample_tokens(
            step_logits.unsqueeze(1),
            temperature=temperature,
        ).squeeze(1)
        return sampled_token_ids, step_logits

    # def _forward_latent_backbone(
    #     self,
    #     *,
    #     position_ids: torch.LongTensor,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     noise_embedding: Optional[torch.Tensor] = None,
    #     target_hidden_states: Optional[torch.Tensor] = None,
    #     past_key_values: Optional[Cache] = None,
    #     use_cache: bool = False,
    #     **kwargs,
    # ) -> torch.Tensor:
    #     hidden_states = noise_embedding
    #     target_hidden_states = self.hidden_norm(self.fc(target_hidden_states))
    #     position_embeddings = self.rotary_emb(hidden_states, position_ids)
    #     for layer in self.latent_layers:
    #         hidden_states = layer(
    #             hidden_states=hidden_states,
    #             target_hidden_states=target_hidden_states,
    #             attention_mask=attention_mask,
    #             position_ids=position_ids,
    #             past_key_value=past_key_values,
    #             use_cache=use_cache,
    #             position_embeddings=position_embeddings,
    #             **kwargs,
    #         )
    #     return self.norm(hidden_states)

    def _forward_backbone(
        self,
        *,
        latent_noise_embedding: Optional[torch.Tensor] = None,
        latent_position_ids: torch.LongTensor,
        latent_attention_mask: Optional[torch.Tensor] = None,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden_states: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        # print(f"latent_noise_embedding: {latent_noise_embedding.shape}", flush=True)
        # print(f"latent_position_ids: {latent_position_ids.shape}", flush=True)
        # print(f"latent_attention_mask: {latent_attention_mask.shape}", flush=True)
        # print(f"position_ids: {position_ids.shape}", flush=True)
        # print(f"attention_mask: {attention_mask.shape}", flush=True)
        # print(f"noise_embedding: {noise_embedding.shape}", flush=True)
        # print(f"target_hidden_states: {target_hidden_states.shape}", flush=True)

        latent_hidden_states = latent_noise_embedding
        target_hidden_states = self.hidden_norm(self.fc(target_hidden_states))
        # print(f"target_hidden_states: {target_hidden_states.shape}", flush=True)
        latent_position_embeddings = self.rotary_emb(latent_hidden_states, latent_position_ids)
        for layer in self.latent_layers:
            latent_hidden_states = layer(
                hidden_states=latent_hidden_states,
                target_hidden_states=target_hidden_states,
                attention_mask=latent_attention_mask,
                position_ids=latent_position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=latent_position_embeddings,
                **kwargs,
            )
        
        hidden_states = noise_embedding
        latent_hidden_states = self.latent_hidden_norm(latent_hidden_states)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden_states=target_hidden_states,
                latent_hidden_states=latent_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)
    
    def forward(
        self,
        input_ids: torch.Tensor, # [bsz, seq_len]
        target_hidden_states: torch.Tensor, # [bsz, seq_len, 5 * hidden_dim]
        loss_mask: torch.Tensor, # [bsz, seq_len]
        target_last_hidden_states: Optional[torch.Tensor] = None, # [bsz, seq_len, hidden_dim]
    ) -> MySpecForwardOutput:
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        # print(f"input_ids.shape: {input_ids.shape}", flush=True)
        # print(f"target_hidden_states.shape: {target_hidden_states.shape}", flush=True)
        # print(f"loss_mask.shape: {loss_mask.shape}", flush=True)
        # print(f"target_last_hidden_states.shape: {target_last_hidden_states.shape}", flush=True)
        # print(f"bsz: {bsz}, seq_len: {seq_len}", flush=True)
        # print(f"block_size: {self.block_size}", flush=True)
        # print(f"num_anchors: {self.num_anchors}", flush=True)
        # print(f"num_latent_tokens: {self.num_latent_tokens}", flush=True)
        # print(f"self.config._attn_implementation:{self.config._attn_implementation}", flush=True)

        anchor_positions, block_keep_mask = sample_anchor_positions(
            seq_len=seq_len,
            loss_mask=loss_mask,
            num_anchors=self.num_anchors,
            device=device,
        ) # [bsz, num_anchors], [bsz, num_anchors]

        latent_noise_embedding = create_latent_noise_embed(
            self.embed_tokens,
            input_ids,
            anchor_positions,
            block_keep_mask,
            latent_token_id=self.latent_token_id,
            num_latent_tokens=self.num_latent_tokens,
        ) # [bsz, num_blocks * num_latent_tokens, hidden_dim]
        
        context_position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1) # [bsz, seq_len]

        latent_position_ids = create_latent_position_ids(
            anchor_positions,
            num_latent_tokens=self.num_latent_tokens,
        ) # [bsz, num_blocks * num_latent_tokens]
        latent_full_position_ids = torch.cat([context_position_ids, latent_position_ids], dim=1) # [bsz, seq_len + num_blocks * num_latent_tokens]
        myspec_latent_attn_mask = create_myspec_latent_attention_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            seq_len=seq_len,
            block_size=self.num_latent_tokens,
            device=device,
        )
        # print(f"latent_noise_embedding.shape: {latent_noise_embedding.shape}", flush=True)
        # print(f"latent_position_ids.shape: {latent_position_ids.shape}", flush=True)
        # print(f"latent_full_position_ids.shape: {latent_full_position_ids.shape}", flush=True)
        # print(f"myspec_latent_attn_mask.shape: {myspec_latent_attn_mask.shape}", flush=True)
        # output_latent_hidden = self._forward_latent_backbone(
        #     position_ids=latent_full_position_ids,
        #     noise_embedding=latent_noise_embedding,
        #     target_hidden_states=target_hidden_states,
        #     attention_mask=myspec_latent_attn_mask,
        # ) # [bsz, num_blocks * num_latent_tokens, hidden_dim]
        # print(f"output_latent_hidden.shape: {output_latent_hidden.shape}", flush=True)
        


        noise_embedding = create_noise_embed(
            self.embed_tokens,
            input_ids,
            anchor_positions,
            block_keep_mask,
            mask_token_id=self.mask_token_id,
            block_size=self.block_size,
        ) # [bsz, num_blocks * block_size, hidden_dim]
        # context_position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1) # [bsz, seq_len]
        draft_position_ids = create_position_ids(anchor_positions, self.block_size) # [bsz, num_blocks * block_size], num_anchors == num_blocks
        # full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1) # [bsz, seq_len + num_blocks * block_size]
        mask_full_position_ids = torch.cat([latent_full_position_ids, draft_position_ids], dim=1) # [bsz, seq_len + num_blocks * block_size]
        myspec_attn_mask = create_myspec_mask_attention_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            seq_len=seq_len,
            block_size=self.block_size,
            num_latent_tokens=self.num_latent_tokens,
            device=device,
        )
        output_hidden = self._forward_backbone(
            latent_noise_embedding=latent_noise_embedding,
            latent_position_ids=latent_full_position_ids,
            latent_attention_mask=myspec_latent_attn_mask,
            position_ids=mask_full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden_states=target_hidden_states,
            attention_mask=myspec_attn_mask,
        ) # [bsz, num_blocks * block_size, hidden_dim]
        num_blocks = anchor_positions.size(1)
        output_hidden_4d = output_hidden.reshape(bsz, num_blocks, self.block_size, -1) # [bsz, num_blocks, block_size, hidden_dim]
        
        # print(f"noise_embedding.shape: {noise_embedding.shape}", flush=True)
        # print(f"myspec_attn_mask.shape: {myspec_attn_mask.shape}", flush=True)
        # print(f"draft_position_ids.shape: {draft_position_ids.shape}", flush=True)
        # print(f"mask_full_position_ids.shape: {mask_full_position_ids.shape}", flush=True)
        # print(f"output_hidden.shape: {output_hidden.shape}", flush=True)
        # print(f"output_hidden_4d.shape: {output_hidden_4d.shape}", flush=True)

        label_offsets = torch.arange(1, self.block_size + 1, device=device).view(
            1, 1, -1
        ) # [1, 1, block_size], [1, 2, ..., block_size]
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets # [bsz, num_blocks, block_size], [anchor_pos + 1, anchor_pos + 2, ..., anchor_pos + block_size]
        safe_label_indices = label_indices.clamp(max=seq_len - 1) # [bsz, num_blocks, block_size]
        safe_label_indices = torch.where(
            block_keep_mask.unsqueeze(-1),
            safe_label_indices,
            torch.zeros_like(safe_label_indices),
        ) # [bsz, num_blocks, block_size], [anchor_pos + 1, anchor_pos + 2, ..., min(anchor_pos + block_size, seq_len - 1)] or [0, 0, ..., 0]
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1), # [bsz, num_blocks, seq_len]
            2,
            safe_label_indices,
        ) # [bsz, num_blocks, block_size], [input_ids[:, anchor_pos + 1], input_ids[:, anchor_pos + 2], ..., input_ids[:, min(anchor_pos + block_size, seq_len - 1)]]
        aligned_target_logits = None
        if target_last_hidden_states is not None:
            target_pred_indices = (safe_label_indices - 1).clamp(min=0) # [bsz, num_blocks, block_size], [anchor_pos, anchor_pos, ..., min(anchor_pos + block_size, seq_len - 1) - 1] or [0, 0, ..., 0]
            aligned_target_hidden = torch.gather(
                target_last_hidden_states.unsqueeze(1).expand(
                    -1,
                    anchor_positions.size(1),
                    -1,
                    -1,
                ), # [bsz, num_blocks, seq_len, hidden_dim]
                2,
                target_pred_indices.unsqueeze(-1).expand(
                    -1,
                    -1,
                    -1,
                    target_last_hidden_states.size(-1),
                ), # [bsz, num_blocks, block_size, hidden_dim]
            ) # [bsz, num_blocks, block_size, hidden_dim]
            aligned_target_logits = self.compute_logits(aligned_target_hidden) # [bsz, num_blocks, block_size, vocab_size]
        eval_mask = build_eval_mask(
            seq_len=seq_len,
            loss_mask=loss_mask, # [bsz, seq_len]
            label_indices=label_indices, # [bsz, num_blocks, block_size], [anchor_pos + 1, anchor_pos + 2, ..., anchor_pos + block_size]
            safe_label_indices=safe_label_indices, # [bsz, num_blocks, block_size], [anchor_pos + 1, anchor_pos + 2, ..., min(anchor_pos + block_size, seq_len - 1)] or [0, 0, ..., 0]
            block_keep_mask=block_keep_mask, # [bsz, num_anchors]
        ) # [bsz, num_blocks, block_size], bool
        anchor_token_ids = torch.gather(
            input_ids, # [bsz, seq_len]
            1,
            anchor_positions, # [bsz, num_blocks]
        ) # [bsz, num_blocks]
        prev_token_ids = torch.cat(
            [anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]],
            dim=-1,
        ) # [bsz, num_blocks, block_size], [input_ids[:, anchor_pos], input_ids[:, anchor_pos + 1], input_ids[:, anchor_pos + 2], ..., input_ids[:, min(anchor_pos + block_size, seq_len - 1)]]
        draft_logits = self.compute_logits(output_hidden).reshape(
            bsz,
            num_blocks,
            self.block_size,
            -1,
        ) # [bsz, num_blocks, block_size, vocab_size]
        if self.markov_head is not None:
            draft_logits = self.markov_head.apply_block_logits(
                draft_logits, # [bsz, num_blocks, block_size, vocab_size]
                token_ids=prev_token_ids, # [bsz, num_blocks, block_size]
                hidden_states=output_hidden_4d, # [bsz, num_blocks, block_size, hidden_dim]
            ) # [bsz, num_blocks, block_size, vocab_size]

        log_sampler_stats(
            seq_len=seq_len,
            loss_mask=loss_mask,
            eval_mask=eval_mask,
            block_keep_mask=block_keep_mask,
            block_size=self.block_size,
            num_anchors=self.num_anchors,
        )

        confidence_pred = None
        if self.confidence_head is not None:
            if self.confidence_head_with_markov:
                prev_embeddings = self.markov_head.get_prev_embeddings(prev_token_ids).to(
                    dtype=output_hidden_4d.dtype
                ) # [bsz, num_blocks, block_size, markov_rank]
                confidence_features = torch.cat(
                    [output_hidden_4d, prev_embeddings],
                    dim=-1,
                ) # [bsz, num_blocks, block_size, hidden_dim + markov_rank]
                confidence_pred = self.confidence_head(confidence_features).float() # [bsz, num_blocks, block_size, 1] -> [bsz, num_blocks, block_size]
            else:
                confidence_pred = self.confidence_head(output_hidden_4d).float()

        return MySpecForwardOutput(
            draft_logits=draft_logits, # [bsz, num_blocks, block_size, vocab_size]
            target_ids=target_ids, # [bsz, num_blocks, block_size], [input_ids[:, anchor_pos + 1], input_ids[:, anchor_pos + 2], ..., input_ids[:, min(anchor_pos + block_size, seq_len - 1)]]
            eval_mask=eval_mask, # [bsz, num_blocks, block_size], bool
            block_keep_mask=block_keep_mask, # [bsz, num_anchors], num_anchors == num_blocks
            confidence_pred=confidence_pred, # [bsz, num_blocks, block_size, 1] -> [bsz, num_blocks, block_size]
            aligned_target_logits=aligned_target_logits, # [bsz, num_blocks, block_size, vocab_size]
        )


__all__ = [
    "Qwen3MySpecModel",
    # "Qwen3MySpecAttention",
    # "Qwen3MySpecDecoderLayer",
]
