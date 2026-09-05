import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from deepspec.modeling.myspec.common import (
    create_latent_cot_block_ids,
    create_myspec_inference_attention_mask,
    create_myspec_inference_latent_attention_mask,
    create_myspec_latent_attention_mask,
)
from deepspec.modeling.myspec.qwen3.modeling import (
    Qwen3DSparkAttention,
    Qwen3MySpecModel,
)


class MySpecTwoStageTest(unittest.TestCase):
    @staticmethod
    def _attention_config():
        return SimpleNamespace(
            hidden_size=2,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=2,
            attention_dropout=0.0,
            attention_bias=False,
            sliding_window=None,
            layer_types=["full_attention"] * 5,
            rms_norm_eps=1e-6,
            latent_cot_token_ids=[151667, 151670, 151670, 151668],
            block_size=7,
            num_latent_layers=2,
        )

    @staticmethod
    def _is_allowed(block_mask, q_idx, kv_idx):
        return bool(
            block_mask.mask_mod(
                torch.tensor(0),
                torch.tensor(0),
                torch.tensor(q_idx),
                torch.tensor(kv_idx),
            )
        )

    def test_flex_attention_receives_compact_gqa_key_value_heads(self):
        config = SimpleNamespace(
            hidden_size=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=2,
            attention_dropout=0.0,
            attention_bias=False,
            sliding_window=None,
            layer_types=["full_attention"] * 5,
            rms_norm_eps=1e-6,
            latent_cot_token_ids=[151667, 151670, 151670, 151668],
            block_size=7,
            num_latent_layers=2,
            _attn_implementation="flex_attention",
        )
        attention = Qwen3DSparkAttention(config, layer_idx=0)
        captured = {}

        def flex_attention_stub(module, query, key, value, attention_mask, **kwargs):
            del module, attention_mask, kwargs
            captured["query_heads"] = query.size(1)
            captured["key_heads"] = key.size(1)
            captured["value_heads"] = value.size(1)
            return query.transpose(1, 2), None

        with patch(
            "deepspec.modeling.myspec.qwen3.modeling.ALL_ATTENTION_FUNCTIONS",
            {"flex_attention": flex_attention_stub},
        ):
            output, _ = attention(
                hidden_states=torch.zeros(1, 5, 8),
                target_hidden_states=torch.zeros(1, 3, 8),
                draft_block_size=5,
                position_embeddings=(
                    torch.ones(1, 8, 2),
                    torch.zeros(1, 8, 2),
                ),
                attention_mask=None,
            )

        self.assertEqual(tuple(output.shape), (1, 5, 8))
        self.assertEqual(captured["query_heads"], 4)
        self.assertEqual(captured["key_heads"], 2)
        self.assertEqual(captured["value_heads"], 2)

    def test_input_block_has_anchor_four_latents_and_seven_masks(self):
        block_ids = create_latent_cot_block_ids(
            torch.tensor([42]),
            latent_cot_token_ids=(151667, 151670, 151670, 151668),
            mask_token_id=151669,
            block_size=7,
        )

        self.assertEqual(
            block_ids.tolist(),
            [[
                42,
                151667,
                151670,
                151670,
                151668,
                151669,
                151669,
                151669,
                151669,
                151669,
                151669,
                151669,
            ]],
        )

    def test_inference_masks_match_two_stage_shapes(self):
        latent_mask = create_myspec_inference_latent_attention_mask(
            batch_size=1,
            context_len=3,
            latent_cot_size=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        full_mask = create_myspec_inference_attention_mask(
            batch_size=1,
            context_len=3,
            latent_cot_size=4,
            block_size=7,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertEqual(tuple(latent_mask.shape), (1, 1, 5, 8))
        self.assertEqual(tuple(full_mask.shape), (1, 1, 12, 15))
        self.assertTrue(torch.equal(latent_mask, torch.zeros_like(latent_mask)))
        # Prefix queries cannot read MASK states in the second stage; MASK
        # queries can read target context and every state in their own block.
        self.assertTrue(
            (full_mask[0, 0, :5, 8:] == torch.finfo(torch.float32).min).all()
        )
        self.assertTrue((full_mask[0, 0, 5:, :] == 0).all())

    def test_training_latent_mask_excludes_other_anchor_blocks(self):
        latent_mask = create_myspec_latent_attention_mask(
            anchor_positions=torch.tensor([[2, 5]]),
            block_keep_mask=torch.tensor([[True, True]]),
            seq_len=8,
            latent_cot_size=4,
            device=torch.device("cpu"),
        )

        # Block zero reads context before anchor position two and its own five
        # prefix states, but neither the anchor's target state nor block one.
        self.assertTrue(self._is_allowed(latent_mask, q_idx=0, kv_idx=1))
        self.assertFalse(self._is_allowed(latent_mask, q_idx=0, kv_idx=2))
        self.assertTrue(self._is_allowed(latent_mask, q_idx=0, kv_idx=8 + 4))
        self.assertFalse(self._is_allowed(latent_mask, q_idx=0, kv_idx=8 + 5))

    def test_attention_uses_separate_kv_projection_per_token_group(self):
        attention = Qwen3DSparkAttention(self._attention_config(), layer_idx=2)
        with torch.no_grad():
            attention.anchor_k_proj.weight.copy_(torch.eye(2))
            attention.latent_k_proj.weight.copy_(torch.eye(2) * 2)
            attention.mask_k_proj.weight.copy_(torch.eye(2) * 3)
            attention.anchor_v_proj.weight.copy_(torch.eye(2) * 4)
            attention.latent_v_proj.weight.copy_(torch.eye(2) * 5)
            attention.mask_v_proj.weight.copy_(torch.eye(2) * 6)

        keys, values = attention._project_draft_kv(
            torch.ones(1, 12, 2),
            draft_block_size=12,
        )

        self.assertTrue(torch.equal(keys[:, :1], torch.ones(1, 1, 2)))
        self.assertTrue(torch.equal(keys[:, 1:5], torch.full((1, 4, 2), 2.0)))
        self.assertTrue(torch.equal(keys[:, 5:], torch.full((1, 7, 2), 3.0)))
        self.assertTrue(torch.equal(values[:, :1], torch.full((1, 1, 2), 4.0)))
        self.assertTrue(torch.equal(values[:, 1:5], torch.full((1, 4, 2), 5.0)))
        self.assertTrue(torch.equal(values[:, 5:], torch.full((1, 7, 2), 6.0)))
        for draft_projection in (
            attention.anchor_k_proj,
            attention.latent_k_proj,
            attention.mask_k_proj,
        ):
            self.assertIsNot(attention.k_proj, draft_projection)
        for draft_projection in (
            attention.anchor_v_proj,
            attention.latent_v_proj,
            attention.mask_v_proj,
        ):
            self.assertIsNot(attention.v_proj, draft_projection)

        early_attention = Qwen3DSparkAttention(
            self._attention_config(),
            layer_idx=0,
        )
        self.assertIsNone(early_attention.mask_k_proj)
        self.assertIsNone(early_attention.mask_v_proj)

    def test_backbone_runs_five_prefix_states_then_all_twelve_states(self):
        calls = []

        class RecordingLayer(nn.Module):
            def __init__(self, layer_idx):
                super().__init__()
                self.layer_idx = layer_idx

            def forward(self, hidden_states, draft_block_size, **kwargs):
                calls.append(
                    (self.layer_idx, hidden_states.size(1), draft_block_size)
                )
                return hidden_states

        class RotaryEmbedding(nn.Module):
            def forward(self, hidden_states, position_ids):
                shape = (*position_ids.shape, hidden_states.size(-1))
                return torch.ones(shape), torch.zeros(shape)

        class StubBackbone(Qwen3MySpecModel):
            def __init__(self):
                nn.Module.__init__(self)
                self.num_anchors = 1
                self.full_block_size = 12
                self.latent_prefix_size = 5
                self.num_latent_layers = 2
                self.layers = nn.ModuleList(
                    [RecordingLayer(idx) for idx in range(5)]
                )
                self.fc = nn.Identity()
                self.hidden_norm = nn.Identity()
                self.norm = nn.Identity()
                self.rotary_emb = RotaryEmbedding()

        output = StubBackbone()._forward_backbone(
            position_ids=torch.arange(15).unsqueeze(0),
            noise_embedding=torch.zeros(1, 12, 2),
            target_hidden_states=torch.zeros(1, 3, 2),
        )

        self.assertEqual(tuple(output.shape), (1, 12, 2))
        self.assertEqual(
            calls,
            [(0, 5, 5), (1, 5, 5), (2, 12, 12), (3, 12, 12), (4, 12, 12)],
        )


if __name__ == "__main__":
    unittest.main()
