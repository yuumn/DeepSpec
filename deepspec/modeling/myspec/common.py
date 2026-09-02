from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask

from deepspec.utils.metrics import add_metric


@dataclass
class MySpecForwardOutput:
    """Outputs for one MySpec training forward.

    Shape symbols:
        batch_size: number of samples in the batch
        seq_len: source sequence length
        num_anchors: sampled anchor blocks per sample
        block_size: number of draft positions per anchor
        vocab_size: vocabulary size

    The sampler keeps anchors whose first draft target is enabled by
    ``loss_mask``. Later slots are supervised only while they remain inside
    ``seq_len`` and form a contiguous enabled prefix. Dummy anchors can still
    appear when a sample has too few valid anchors; they are masked out by
    ``block_keep_mask`` and ``eval_mask``.
    """

    # [batch_size, num_anchors, block_size, vocab_size]
    draft_logits: torch.Tensor
    # [batch_size, num_anchors, block_size]
    target_ids: torch.Tensor
    # [batch_size, num_anchors, block_size]
    eval_mask: torch.Tensor
    # [batch_size, num_anchors]
    block_keep_mask: torch.Tensor
    # [batch_size, num_anchors, block_size]
    confidence_pred: Optional[torch.Tensor] = None
    # [batch_size, num_anchors, block_size, vocab_size]
    aligned_target_logits: Optional[torch.Tensor] = None


class AcceptRatePredictor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(int(input_dim), 1)

    def forward(self, features):
        return self.proj(features).squeeze(-1)


def extract_context_feature(hidden_states, layer_ids):
    return torch.cat(
        [hidden_states[0 if layer_id == -1 else layer_id + 1] for layer_id in layer_ids],
        dim=-1,
    )


def validate_target_layer_ids(layer_ids, num_target_layers: int):
    layer_ids = [int(layer_id) for layer_id in layer_ids]
    assert layer_ids, "target_layer_ids must not be empty."
    start = 0
    end = int(num_target_layers) - 1
    previous = None
    for layer_id in layer_ids:
        assert layer_id == -1 or start <= layer_id <= end, (
            f"target_layer_id {layer_id} is out of range {{-1}} U [{start}, {end}] "
            f"for num_target_layers={num_target_layers}. "
            "-1 denotes the embedding output."
        )
        assert previous is None or layer_id > previous, (
            "target_layer_ids must be strictly increasing."
        )
        previous = layer_id
    return layer_ids


def create_myspec_attention_mask(
    *,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    seq_len: int,
    block_size: int,
    latent_cot_size: int,
    device: torch.device,
):
    latent_prefix_size = 1 + int(latent_cot_size)  # current token + latent CoT
    full_block_size = latent_prefix_size + int(block_size)

    def myspec_mask_mod(b, h, q_idx, kv_idx):
        del h
        q_block_id = q_idx // full_block_size
        anchor_pos = anchor_positions[b, q_block_id]
        is_context = kv_idx < seq_len
        mask_context = is_context & (kv_idx < anchor_pos)
        is_draft = kv_idx >= seq_len
        kv_block_id = (kv_idx - seq_len) // full_block_size
        is_same_block = is_draft & (q_block_id == kv_block_id)
        q_is_latent_prefix = (q_idx % full_block_size) < latent_prefix_size
        kv_is_latent_prefix = (
            (kv_idx - seq_len) % full_block_size
        ) < latent_prefix_size
        mask_block = is_same_block & (
            (~q_is_latent_prefix) | kv_is_latent_prefix
        )
        is_valid_block = block_keep_mask[b, q_block_id]
        return (mask_context | mask_block) & is_valid_block

    bsz, num_blocks = anchor_positions.shape
    return create_block_mask(
        myspec_mask_mod,
        B=bsz,
        H=None,
        Q_LEN=num_blocks * full_block_size,
        KV_LEN=seq_len + num_blocks * full_block_size,
        device=device,
    )


def create_myspec_inference_attention_mask(
    *,
    batch_size: int,
    context_len: int,
    block_size: int,
    latent_cot_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    latent_prefix_size = 1 + int(latent_cot_size)
    full_block_size = latent_prefix_size + int(block_size)
    allowed = torch.ones(
        batch_size,
        1,
        full_block_size,
        context_len + full_block_size,
        dtype=torch.bool,
        device=device,
    )
    allowed[:, :, :latent_prefix_size, context_len + latent_prefix_size :] = False
    attention_mask = torch.zeros(allowed.shape, dtype=dtype, device=device)
    return attention_mask.masked_fill(~allowed, torch.finfo(dtype).min)


def create_latent_cot_block_ids(
    current_token_ids: torch.Tensor,
    *,
    latent_cot_token_ids: tuple[int, ...],
    mask_token_id: int,
    block_size: int,
) -> torch.Tensor:
    suffix_ids = torch.tensor(
        [*latent_cot_token_ids, *([int(mask_token_id)] * int(block_size))],
        dtype=current_token_ids.dtype,
        device=current_token_ids.device,
    )
    suffix_ids = suffix_ids.view(*([1] * current_token_ids.ndim), -1).expand(
        *current_token_ids.shape,
        -1,
    )
    return torch.cat([current_token_ids.unsqueeze(-1), suffix_ids], dim=-1)


def build_anchor_candidate_mask(
    *,
    seq_len: int,
    loss_mask: torch.Tensor, # [bsz, seq_len]
) -> torch.Tensor:
    num_candidates = max(seq_len - 1, 0)
    if num_candidates == 0:
        return loss_mask[:, :0].bool()

    anchor_valid = loss_mask[:, :num_candidates] > 0.5
    first_target_valid = loss_mask[:, 1 : num_candidates + 1] > 0.5
    return anchor_valid & first_target_valid # [bsz, seq_len - 1], bool


def sample_anchor_positions(
    *,
    seq_len: int,
    loss_mask: torch.Tensor, # [bsz, seq_len]
    num_anchors: int, # 512
    device: torch.device, 
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = build_anchor_candidate_mask(
        seq_len=seq_len,
        loss_mask=loss_mask,
    ) # [bsz, seq_len - 1], bool
    valid_counts = valid.sum(dim=1) # [bsz]
    bsz = loss_mask.shape[0]
    num_candidates = valid.shape[1]
    max_n = int(num_anchors)
    if num_candidates == 0:
        anchors = torch.zeros(bsz, max_n, dtype=torch.long, device=device)
        keep_mask = torch.zeros(bsz, max_n, dtype=torch.bool, device=device)
        return anchors, keep_mask

    indices = torch.arange(num_candidates, device=device).unsqueeze(0).expand(
        bsz,
        -1,
    ) # [bsz, seq_len - 1]
    masked_indices = torch.where(
        valid,
        indices,
        torch.full_like(indices, seq_len + 1),
    ) # [bsz, seq_len - 1]
    random_vals = torch.rand(bsz, num_candidates, device=device) # [bsz, seq_len - 1]
    random_vals = torch.where(valid, random_vals, torch.full_like(random_vals, 2.0)) # [bsz, seq_len - 1]
    _, sorted_idx = random_vals.sort(dim=1) # [bsz, seq_len - 1]
    gathered = torch.gather(masked_indices, 1, sorted_idx) # [bsz, seq_len - 1]
    if num_candidates < max_n:
        pad = torch.full(
            (bsz, max_n - num_candidates),
            seq_len + 1,
            dtype=gathered.dtype,
            device=device,
        )
        gathered = torch.cat([gathered, pad], dim=1)
    anchors = gathered[:, :max_n].sort(dim=1).values # [bsz, num_anchors]
    keep_mask = torch.arange(max_n, device=device).unsqueeze(0) < (
        valid_counts.unsqueeze(1).clamp(max=max_n)
    ) # [bsz, num_anchors]
    anchors = torch.where(keep_mask, anchors, torch.zeros_like(anchors)) # [bsz, num_anchors]
    return anchors, keep_mask


def build_eval_mask(
    *,
    seq_len: int,
    loss_mask: torch.Tensor, # [bsz, seq_len]
    label_indices: torch.Tensor, # [bsz, num_blocks, block_size], [anchor_pos + 1, anchor_pos + 2, ..., anchor_pos + block_size]
    safe_label_indices: torch.Tensor, # [bsz, num_blocks, block_size], [anchor_pos + 1, anchor_pos + 2, ..., min(anchor_pos + block_size, seq_len - 1)] or [0, 0, ..., 0]
    block_keep_mask: torch.Tensor, # [bsz, num_anchors], num_blocks == num_anchors
) -> torch.Tensor:
    target_valid = label_indices < seq_len # [bsz, num_blocks, block_size]
    target_loss_mask = torch.gather(
        loss_mask.unsqueeze(1).expand(-1, label_indices.size(1), -1), # [bsz, num_blocks, seq_len]
        2,
        safe_label_indices,
    ) # [bsz, num_blocks, block_size]
    eval_mask = target_valid & (target_loss_mask > 0.5) # [bsz, num_blocks, block_size]
    eval_mask = eval_mask & block_keep_mask.unsqueeze(-1) # [bsz, num_blocks, block_size]
    return eval_mask.to(torch.int32).cumprod(dim=-1).bool() # [bsz, num_blocks, block_size], bool


@torch.no_grad()
def log_sampler_stats(
    *,
    seq_len: int,
    loss_mask: torch.Tensor,
    block_keep_mask: torch.Tensor,
    eval_mask: torch.Tensor,
    block_size: int,
    num_anchors: int,
) -> None:
    valid_anchor_mask = build_anchor_candidate_mask(
        seq_len=seq_len,
        loss_mask=loss_mask,
    ) # [bsz, seq_len - 1], bool
    valid_anchor_counts = valid_anchor_mask.sum(dim=1).to(torch.float32)
    valid_anchor_ratios = valid_anchor_counts / max(float(seq_len), 1.0)
    sampled_anchor_counts = block_keep_mask.sum(dim=1).to(torch.float32)
    sampled_anchor_ratios = sampled_anchor_counts / max(float(num_anchors), 1.0)
    sample_count = loss_mask.new_tensor(float(loss_mask.shape[0]), dtype=torch.float32)
    add_metric(
        "valid_anchors_abs",
        valid_anchor_counts.sum(),
        den=sample_count,
        tag="train",
    )
    add_metric(
        "valid_anchors_ratio",
        valid_anchor_ratios.sum(),
        den=sample_count,
        tag="train",
    )
    add_metric(
        "sampled_anchors_abs",
        sampled_anchor_counts.sum(),
        den=sample_count,
        tag="train",
    )
    add_metric(
        "sampled_anchors_ratio",
        sampled_anchor_ratios.sum(),
        den=sample_count,
        tag="train",
    )
    block_supervised_tokens = eval_mask.to(torch.float32).sum(dim=(1, 2)) / (
        sampled_anchor_counts.clamp_min(1.0)
    )
    add_metric(
        "block_supervised_tokens_abs",
        block_supervised_tokens.sum(),
        den=sample_count,
        tag="train",
    )
    add_metric(
        "block_supervised_tokens_ratio",
        (block_supervised_tokens / float(block_size)).sum(),
        den=sample_count,
        tag="train",
    )


def create_position_ids(
    anchor_positions: torch.Tensor, # [bsz, num_anchors]
    block_size: int,
) -> torch.Tensor:
    bsz, num_blocks = anchor_positions.shape
    device = anchor_positions.device
    offsets = torch.arange(block_size, device=device).view(1, 1, -1)
    return (anchor_positions.unsqueeze(-1) + offsets).view(
        bsz,
        num_blocks * block_size,
    )


__all__ = [
    "MySpecForwardOutput",
    "AcceptRatePredictor",
    "extract_context_feature",
    "validate_target_layer_ids",
    "create_myspec_attention_mask",
    "create_myspec_inference_attention_mask",
    "create_latent_cot_block_ids",
    "build_anchor_candidate_mask",
    "sample_anchor_positions",
    "build_eval_mask",
    "log_sampler_stats",
    "create_position_ids",
]
