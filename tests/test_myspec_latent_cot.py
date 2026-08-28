import unittest

import torch

from deepspec.modeling.myspec.common import (
    MySpecForwardOutput,
    create_latent_attention_mask,
    create_latent_draft_attention_mask,
    create_latent_position_ids,
    create_position_ids,
)
from deepspec.modeling.myspec.loss import (
    _compute_accept_rate_3d,
    _compute_local_prefix_accept_term,
)


def _allowed(block_mask, q_idx: int, kv_idx: int) -> bool:
    return bool(
        block_mask.mask_mod(
            torch.tensor(0),
            torch.tensor(0),
            torch.tensor(q_idx),
            torch.tensor(kv_idx),
        )
    )


class LatentCoTMaskTest(unittest.TestCase):
    def setUp(self):
        self.anchors = torch.tensor([[2, 5]])
        self.keep = torch.tensor([[True, True]])
        self.seq_len = 8
        self.num_latent_tokens = 3
        self.block_size = 2

    def test_latent_mask_is_causal_and_isolates_anchors(self):
        mask = create_latent_attention_mask(
            anchor_positions=self.anchors,
            block_keep_mask=self.keep,
            seq_len=self.seq_len,
            num_latent_tokens=self.num_latent_tokens,
            causal=True,
            device=torch.device("cpu"),
        )

        # Anchor 0 sees context [0, 2), but not its anchor/future context.
        self.assertTrue(_allowed(mask, q_idx=2, kv_idx=1))
        self.assertFalse(_allowed(mask, q_idx=2, kv_idx=2))

        # Its third latent slot sees only slots 0..2 from its own block.
        latent_start = self.seq_len
        self.assertTrue(_allowed(mask, q_idx=2, kv_idx=latent_start + 2))
        self.assertFalse(
            _allowed(
                mask,
                q_idx=2,
                kv_idx=latent_start + self.num_latent_tokens,
            )
        )

        # The first latent slot cannot read later slots in a causal chain.
        self.assertFalse(_allowed(mask, q_idx=0, kv_idx=latent_start + 1))

    def test_draft_mask_reads_only_its_own_latent_and_draft_blocks(self):
        mask = create_latent_draft_attention_mask(
            anchor_positions=self.anchors,
            block_keep_mask=self.keep,
            seq_len=self.seq_len,
            num_latent_tokens=self.num_latent_tokens,
            block_size=self.block_size,
            device=torch.device("cpu"),
        )
        latent_start = self.seq_len
        draft_start = self.seq_len + 2 * self.num_latent_tokens

        # Query 0 belongs to anchor 0.
        self.assertTrue(_allowed(mask, q_idx=0, kv_idx=1))
        self.assertFalse(_allowed(mask, q_idx=0, kv_idx=2))
        self.assertTrue(_allowed(mask, q_idx=0, kv_idx=latent_start + 2))
        self.assertFalse(
            _allowed(
                mask,
                q_idx=0,
                kv_idx=latent_start + self.num_latent_tokens,
            )
        )
        self.assertTrue(_allowed(mask, q_idx=0, kv_idx=draft_start + 1))
        self.assertFalse(
            _allowed(mask, q_idx=0, kv_idx=draft_start + self.block_size)
        )

    def test_latent_positions_do_not_shift_real_draft_positions(self):
        latent_positions = create_latent_position_ids(
            self.anchors,
            self.num_latent_tokens,
        )
        draft_positions = create_position_ids(self.anchors, self.block_size)
        self.assertEqual(latent_positions.tolist(), [[2, 2, 2, 5, 5, 5]])
        self.assertEqual(draft_positions.tolist(), [[2, 3, 5, 6]])


class PrefixAcceptLossTest(unittest.TestCase):
    def _outputs(self, draft_logits):
        return MySpecForwardOutput(
            draft_logits=draft_logits,
            target_ids=torch.zeros(1, 1, 2, dtype=torch.long),
            eval_mask=torch.ones(1, 1, 2, dtype=torch.bool),
            block_keep_mask=torch.ones(1, 1, dtype=torch.bool),
        )

    def test_prefix_loss_is_zero_for_matching_distributions(self):
        logits = torch.tensor([[[[2.0, -2.0], [1.0, -1.0]]]])
        outputs = self._outputs(logits)
        accept_rate = _compute_accept_rate_3d(
            outputs=outputs,
            aligned_target_logits=logits.clone(),
        )
        numerator, denominator = _compute_local_prefix_accept_term(
            outputs=outputs,
            accept_rate_3d=accept_rate,
        )
        self.assertAlmostEqual(float(numerator / denominator), 0.0, places=6)

    def test_prefix_loss_penalizes_mismatched_distributions(self):
        draft_logits = torch.tensor([[[[10.0, -10.0], [10.0, -10.0]]]])
        target_logits = -draft_logits
        outputs = self._outputs(draft_logits)
        accept_rate = _compute_accept_rate_3d(
            outputs=outputs,
            aligned_target_logits=target_logits,
        )
        numerator, denominator = _compute_local_prefix_accept_term(
            outputs=outputs,
            accept_rate_3d=accept_rate,
        )
        self.assertGreater(float(numerator / denominator), 0.99)


if __name__ == "__main__":
    unittest.main()
