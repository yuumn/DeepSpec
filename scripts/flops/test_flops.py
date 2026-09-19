#!/usr/bin/env python3
"""Unit tests for the analytical FLOP equations and config resolution."""

from __future__ import annotations

import unittest
from pathlib import Path

from flops_core import (
    DraftArchitecture,
    ModelDimensions,
    architecture_from_config,
    comparison_rows,
    decoder_layer_inference_linear_flops,
    decoder_layer_training_linear_flops,
    inference_breakdown,
    output_heads_inference_breakdown,
    output_heads_training_breakdown,
    total,
    training_breakdown,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


class FlopsCoreTest(unittest.TestCase):
    def test_configs_are_the_requested_experiments(self) -> None:
        myspec = architecture_from_config(
            REPO_ROOT / "config/myspec/myspec_qwen3_4b_eager.py",
            name="MySpec",
            kind="myspec",
        )
        markov_myspec = architecture_from_config(
            REPO_ROOT / "config/myspec/myspec_qwen3_4b.py",
            name="MySpec-Markov-Conf",
            kind="myspec",
        )
        dspark = architecture_from_config(
            REPO_ROOT / "config/dspark/dspark_qwen3_4b.py",
            name="DSpark",
            kind="dspark",
        )
        dflash = architecture_from_config(
            REPO_ROOT.parent / "DeepSpec/config/dflash/dflash_qwen3_4b.py",
            name="DFlash",
            kind="dspark",
        )

        self.assertEqual(myspec.num_latent_tokens, 4)
        self.assertEqual((myspec.num_latent_layers, myspec.num_draft_layers), (2, 3))
        self.assertFalse(myspec.has_markov)
        self.assertFalse(myspec.confidence_enabled)
        self.assertEqual(markov_myspec.markov_rank, 256)
        self.assertTrue(markov_myspec.confidence_enabled)
        self.assertTrue(markov_myspec.confidence_with_markov)
        self.assertEqual(dspark.num_draft_layers, 5)
        self.assertEqual(dspark.markov_rank, 256)
        self.assertTrue(dspark.confidence_enabled)
        self.assertEqual(dflash.num_draft_layers, 5)
        self.assertEqual(dflash.markov_rank, 0)
        self.assertFalse(dflash.confidence_enabled)

    def test_forward_decoder_formula_matches_enumerated_gemms(self) -> None:
        dims = ModelDimensions(
            hidden_size=12,
            intermediate_size=20,
            num_attention_heads=3,
            num_key_value_heads=1,
            head_dim=4,
            vocab_size=31,
            num_target_layers=2,
        )
        q = 7
        external = (11, 5)
        expected_macs = (
            q * 12 * 12  # Q
            + q * 12 * 12  # O
            + 2 * q * 12 * 4  # own K/V
            + 2 * (11 + 5) * 12 * 4  # external K/V pairs
            + 3 * q * 12 * 20  # gated MLP
        )
        actual = decoder_layer_inference_linear_flops(
            query_tokens=q,
            external_kv_tokens=external,
            dims=dims,
        )
        self.assertEqual(actual, 2 * expected_macs)

    def test_first_stack_layer_omits_only_frozen_input_gradients(self) -> None:
        dims = ModelDimensions(
            hidden_size=12,
            intermediate_size=20,
            num_attention_heads=3,
            num_key_value_heads=1,
            head_dim=4,
            vocab_size=31,
            num_target_layers=2,
        )
        q = 7
        regular = decoder_layer_training_linear_flops(
            query_tokens=q,
            external_kv_tokens=(11,),
            dims=dims,
            first_layer_in_stack=False,
        )
        first = decoder_layer_training_linear_flops(
            query_tokens=q,
            external_kv_tokens=(11,),
            dims=dims,
            first_layer_in_stack=True,
        )
        omitted_input_gradients = 2 * q * 12 * 12 + 4 * q * 12 * 4
        self.assertEqual(regular - first, omitted_input_gradients)

    def test_head_paths_match_freeze_and_auxiliary_head_semantics(self) -> None:
        dims = ModelDimensions(
            hidden_size=12,
            intermediate_size=20,
            num_attention_heads=3,
            num_key_value_heads=1,
            head_dim=4,
            vocab_size=31,
            num_target_layers=2,
        )
        architecture = DraftArchitecture(
            name="aux",
            kind="dspark",
            num_feature_layers=5,
            num_anchors=2,
            block_size=3,
            num_draft_layers=1,
            markov_rank=5,
            confidence_enabled=True,
            confidence_with_markov=True,
        )
        train = output_heads_training_breakdown(6, dims, architecture)
        infer = output_heads_inference_breakdown(3, dims, architecture)
        self.assertEqual(train["draft LM head"], 4 * 6 * 12 * 31)
        self.assertEqual(train["aligned target LM head"], 2 * 6 * 12 * 31)
        self.assertEqual(train["Markov head"], 6 * 6 * 5 * 31)
        self.assertEqual(train["confidence head"], 6 * 6 * (12 + 5))
        self.assertEqual(infer["Markov head"], 2 * 3 * 5 * 31)
        self.assertEqual(infer["confidence head"], 2 * 3 * (12 + 5))

    def test_padded_anchor_only_loses_logical_attention_pairs(self) -> None:
        dims = ModelDimensions(
            hidden_size=12,
            intermediate_size=20,
            num_attention_heads=3,
            num_key_value_heads=1,
            head_dim=4,
            vocab_size=31,
            num_target_layers=2,
        )
        architecture = DraftArchitecture(
            name="tiny",
            kind="dspark",
            num_feature_layers=2,
            num_anchors=8,
            block_size=3,
            num_draft_layers=1,
        )
        breakdown = training_breakdown(4, dims, architecture)
        # S=4 has three real anchors (positions 0,1,2), mean=1.  Each of their
        # B=3 queries sees mean_anchor+B=4 keys.
        expected_pairs = 3 * 3 * 4
        self.assertEqual(
            breakdown["draft attention"],
            12 * expected_pairs * dims.query_projection_size,
        )
        # Linear/head paths still execute all num_anchors padded blocks.
        self.assertEqual(architecture.training_draft_tokens, 24)

    def test_configured_candidates_reduce_flops_at_all_reported_contexts(self) -> None:
        dims = ModelDimensions()
        myspec = architecture_from_config(
            REPO_ROOT / "config/myspec/myspec_qwen3_4b_eager.py",
            name="MySpec",
            kind="myspec",
        )
        dflash = architecture_from_config(
            REPO_ROOT.parent / "DeepSpec/config/dflash/dflash_qwen3_4b.py",
            name="DFlash",
            kind="dspark",
        )
        rows = comparison_rows(
            contexts=(512, 1024, 2048, 4096),
            dims=dims,
            candidate=myspec,
            baseline=dflash,
            new_context_tokens=8,
        )
        for row in rows:
            self.assertGreater(row["train_reduction"], 0)
            self.assertGreater(row["infer_reduction"], 0)
            self.assertGreater(row["first_reduction"], 0)
            self.assertGreater(row["e2e_reduction"], 0)

    def test_breakdowns_sum_to_positive_values(self) -> None:
        architecture = DraftArchitecture(
            name="tiny",
            kind="myspec",
            num_feature_layers=2,
            num_anchors=3,
            block_size=2,
            num_draft_layers=2,
            num_latent_layers=1,
            num_latent_tokens=1,
        )
        dims = ModelDimensions(
            hidden_size=12,
            intermediate_size=20,
            num_attention_heads=3,
            num_key_value_heads=1,
            head_dim=4,
            vocab_size=31,
            num_target_layers=2,
        )
        self.assertGreater(total(training_breakdown(8, dims, architecture)), 0)
        self.assertGreater(total(inference_breakdown(8, 2, dims, architecture)), 0)


if __name__ == "__main__":
    unittest.main()
