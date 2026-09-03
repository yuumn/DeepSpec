import unittest

import torch

from deepspec.modeling.myspec.common import (
    create_myspec_inference_position_ids,
    create_myspec_position_ids,
)


class MySpecPositionIdsTest(unittest.TestCase):
    def test_training_positions_use_separate_latent_and_prediction_offsets(self):
        anchor_positions = torch.tensor([[2, 5]])

        position_ids = create_myspec_position_ids(
            anchor_positions,
            latent_cot_size=4,
            block_size=7,
        )

        self.assertEqual(
            position_ids.tolist(),
            [[
                2, 3, 4, 5, 6, 3, 4, 5, 6, 7, 8, 9,
                5, 6, 7, 8, 9, 6, 7, 8, 9, 10, 11, 12,
            ]],
        )

    def test_inference_positions_preserve_uncached_context(self):
        position_ids = create_myspec_inference_position_ids(
            past_len=2,
            anchor_position=5,
            batch_size=2,
            latent_cot_size=4,
            block_size=7,
            device=torch.device("cpu"),
        )

        expected = [
            2, 3, 4,
            5, 6, 7, 8, 9,
            6, 7, 8, 9, 10, 11, 12,
        ]
        self.assertEqual(position_ids.tolist(), [expected, expected])


if __name__ == "__main__":
    unittest.main()
