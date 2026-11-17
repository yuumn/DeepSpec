"""Full-context draft backbone with past_key_values=None and use_cache=False."""
import torch


def forward_without_cache(model, *, context, ids, latent_ids, positions, start):
    if latent_ids is None:
        return model._forward_backbone(
            target_hidden_states=context, noise_embedding=model.embed_tokens(ids),
            position_ids=positions[:, :start+ids.shape[1]], attention_mask=None,
            past_key_values=None, use_cache=False, is_causal=False)
    prefix_positions=positions[:, :start]
    latent_positions=positions[:, start:start+model.num_latent_tokens]
    mask_positions=positions[:, start:start+ids.shape[1]]
    return model._forward_backbone(
        target_hidden_states=context,
        latent_noise_embedding=model.embed_tokens(latent_ids),
        latent_position_ids=torch.cat([prefix_positions,latent_positions],dim=1),
        noise_embedding=model.embed_tokens(ids),
        position_ids=torch.cat([prefix_positions,latent_positions,mask_positions],dim=1),
        attention_mask=None, past_key_values=None, use_cache=False, is_causal=False)
