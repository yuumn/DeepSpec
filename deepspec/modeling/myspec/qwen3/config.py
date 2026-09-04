import copy

from deepspec.modeling.myspec.common import validate_target_layer_ids


TRAIN_ATTN_IMPLEMENTATION = "flex_attention"


def build_draft_config(
    target_config,
    model_args,
):
    num_target_layers = int(target_config.num_hidden_layers)
    num_draft_layers = int(model_args.num_draft_layers)
    num_latent_layers = int(model_args.num_latent_layers)
    assert 0 < num_latent_layers < num_draft_layers, (
        "num_latent_layers must split the draft stack into two non-empty stages: "
        f"got num_latent_layers={num_latent_layers}, "
        f"num_draft_layers={num_draft_layers}."
    )
    layer_types = ["full_attention"] * num_draft_layers
    assert "target_layer_ids" in model_args, "target_layer_ids must be provided."
    target_layer_ids = validate_target_layer_ids(
        model_args.target_layer_ids,
        num_target_layers,
    )

    confidence_head_alpha = float(model_args.confidence_head_alpha)
    assert confidence_head_alpha >= 0.0
    enable_confidence_head = confidence_head_alpha > 0.0
    if enable_confidence_head:
        assert "confidence_head_with_markov" in model_args, (
            "confidence_head_with_markov must be provided when "
            "confidence_head_alpha > 0."
        )
    markov_rank = int(model_args.markov_rank)
    assert markov_rank >= 0, f"markov_rank must be >= 0, got {markov_rank}"
    if markov_rank > 0:
        assert "markov_head_type" in model_args, (
            "markov_head_type must be provided when markov_rank > 0."
        )
    latent_cot_token_ids = [
        int(token_id) for token_id in model_args.latent_cot_token_ids
    ]
    assert len(latent_cot_token_ids) == 4, (
        "Exactly four latent CoT tokens are required."
    )
    assert all(
        0 <= token_id < target_config.vocab_size
        for token_id in latent_cot_token_ids
    )

    draft_config = copy.deepcopy(target_config)
    draft_config.architectures = ["Qwen3MySpecModel"]
    draft_config.num_target_layers = num_target_layers
    draft_config.num_hidden_layers = num_draft_layers
    draft_config.num_latent_layers = num_latent_layers
    draft_config.block_size = int(model_args.block_size)
    draft_config.tie_word_embeddings = False
    draft_config.layer_types = layer_types
    draft_config._attn_implementation = TRAIN_ATTN_IMPLEMENTATION
    draft_config.mask_token_id = int(model_args.mask_token_id)
    draft_config.latent_cot_token_ids = latent_cot_token_ids
    draft_config.target_layer_ids = target_layer_ids
    draft_config.num_anchors = int(model_args.num_anchors)
    draft_config.enable_confidence_head = enable_confidence_head
    if enable_confidence_head:
        draft_config.confidence_head_with_markov = bool(
            model_args.confidence_head_with_markov
        )
    draft_config.markov_rank = markov_rank
    if markov_rank > 0:
        draft_config.markov_head_type = str(model_args.markov_head_type)
    return draft_config


__all__ = [
    "build_draft_config",
]
