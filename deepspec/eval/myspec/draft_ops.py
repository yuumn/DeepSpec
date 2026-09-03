from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import DynamicCache

from deepspec.eval.base_evaluator import DraftProposal
from deepspec.utils.sampling import logits_to_probs
from deepspec.modeling.myspec.common import (
    create_myspec_inference_attention_mask,
    create_myspec_inference_position_ids,
)
from deepspec.modeling.myspec.qwen3 import Qwen3MySpecModel


MySpecModel = Qwen3MySpecModel


@dataclass
class MySpecDraftProposal(DraftProposal):
    confidence_logits: torch.Tensor | None = None


def forward_myspec_draft_block(
    model: MySpecModel,
    *,
    draft_input_ids: torch.Tensor,
    past_key_values_draft: DynamicCache,
    target_hidden_states: torch.Tensor,
    start: int,
    block_size: int,
) -> torch.Tensor:
    assert block_size == model.block_size
    assert draft_input_ids.size(1) == model.full_block_size
    past_len = past_key_values_draft.get_seq_length()
    assert target_hidden_states.size(1) == start - past_len
    draft_position_ids = create_myspec_inference_position_ids(
        past_len=past_len,
        anchor_position=start,
        batch_size=draft_input_ids.size(0),
        latent_cot_size=model.latent_cot_size,
        block_size=block_size,
        device=draft_input_ids.device,
    )
    draft_embedding = model.embed_tokens(draft_input_ids)
    attention_mask = create_myspec_inference_attention_mask(
        batch_size=draft_input_ids.size(0),
        context_len=start,
        block_size=block_size,
        latent_cot_size=model.latent_cot_size,
        device=draft_input_ids.device,
        dtype=draft_embedding.dtype,
    )
    block_hidden = model._forward_backbone(
        target_hidden_states=target_hidden_states,
        noise_embedding=draft_embedding,
        position_ids=draft_position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values_draft,
        use_cache=True,
        is_causal=False,
    )
    past_key_values_draft.crop(start)
    return block_hidden


def _empty_myspec_proposal(draft_input_ids: torch.Tensor) -> MySpecDraftProposal:
    return MySpecDraftProposal(
        draft_token_count=0,
        verify_input_ids=draft_input_ids[:, :1],
        draft_probs=None,
        confidence_logits=None,
    )


def _predict_confidence_logits(
    model: MySpecModel,
    *,
    proposal_hidden_states: torch.Tensor,
    draft_input_ids: torch.Tensor,
    sampled_tokens: torch.Tensor,
    block_size: int,
) -> torch.Tensor | None:
    prev_token_ids = torch.cat(
        [draft_input_ids[:, :1], sampled_tokens[:, :-1]],
        dim=1,
    )
    confidence_pred = model.predict_confidence_step(
        proposal_hidden_states,
        prev_token_ids=prev_token_ids,
    )
    if confidence_pred is None:
        return None
    return confidence_pred.float().reshape(
        confidence_pred.shape[0],
        block_size,
        -1,
    )[:, :, 0]


def _confident_prefix_length(
    confidence_logits: torch.Tensor,
    *,
    block_size: int,
    threshold: float,
) -> int:
    if threshold <= 0.0:
        return int(block_size)
    below_threshold = confidence_logits.sigmoid() < threshold
    if not bool(below_threshold[0].any().item()):
        return int(block_size)
    return int(torch.nonzero(below_threshold[0], as_tuple=False)[0].item())


def build_myspec_proposal(
    model: MySpecModel,
    *,
    draft_input_ids: torch.Tensor,
    block_hidden: torch.Tensor,
    block_size: int,
    temperature: float,
    confidence_threshold: float,
) -> MySpecDraftProposal:
    assert draft_input_ids.size(0) == 1, "build_myspec_proposal requires batch_size=1"
    proposal_hidden_states = block_hidden[
        :, model.latent_prefix_size : model.latent_prefix_size + block_size, :
    ]
    base_draft_logits = model.compute_logits(proposal_hidden_states)
    sampled_tokens, draft_logits = model.sample_draft_tokens(
        base_draft_logits,
        first_prev_token_ids=draft_input_ids[:, 0],
        temperature=temperature,
        hidden_states=proposal_hidden_states,
    )

    proposal_draft_tokens = int(block_size)
    confidence_logits = None
    if model.confidence_head is not None:
        confidence_logits = _predict_confidence_logits(
            model,
            proposal_hidden_states=proposal_hidden_states,
            draft_input_ids=draft_input_ids,
            sampled_tokens=sampled_tokens,
            block_size=block_size,
        )
        if confidence_logits is None:
            return _empty_myspec_proposal(draft_input_ids)
        proposal_draft_tokens = _confident_prefix_length(
            confidence_logits,
            block_size=block_size,
            threshold=float(confidence_threshold),
        )

    if proposal_draft_tokens == 0:
        return _empty_myspec_proposal(draft_input_ids)

    verify_input_ids = torch.cat(
        [draft_input_ids[:, :1], sampled_tokens[:, :proposal_draft_tokens]],
        dim=1,
    )
    draft_probs = logits_to_probs(
        draft_logits[:, :proposal_draft_tokens, :],
        temperature,
    )
    return MySpecDraftProposal(
        draft_token_count=proposal_draft_tokens,
        verify_input_ids=verify_input_ids,
        draft_probs=draft_probs,
        confidence_logits=(
            confidence_logits[:, :proposal_draft_tokens]
            if confidence_logits is not None
            else None
        ),
    )
