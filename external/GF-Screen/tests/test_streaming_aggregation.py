import unittest

import torch

from models.GF_Screen_RL import (
    accumulate_segmentation_with_coords,
    aggregate_segmentation_with_coords,
    finalize_segmentation_aggregation,
)


class StreamingAggregationTests(unittest.TestCase):
    def _compare_streamed(self, mode):
        generator = torch.Generator().manual_seed(20260820)
        patches = torch.rand((5, 2, 4, 4, 3), generator=generator)
        coords = torch.tensor(
            [[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 1], [1, 1, 1]]
        )
        expected = aggregate_segmentation_with_coords(
            torch.zeros((1, 2, 7, 7, 5)), patches, coords, mode=mode
        )

        output_sum = torch.zeros((1, 2, 7, 7, 5))
        count_map = torch.zeros_like(output_sum)
        gaussian_map = None
        for start in range(0, len(patches), 2):
            gaussian_map = accumulate_segmentation_with_coords(
                output_sum,
                count_map,
                patches[start:start + 2],
                coords[start:start + 2],
                mode=mode,
                gaussian_map=gaussian_map,
            )
        actual = finalize_segmentation_aggregation(output_sum, count_map)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_uniform_matches_complete_batch(self):
        self._compare_streamed("uniform")

    def test_gaussian_matches_complete_batch(self):
        self._compare_streamed("gaussian")

    def test_uncovered_voxels_remain_zero(self):
        patches = torch.ones((1, 2, 2, 2, 2))
        output = aggregate_segmentation_with_coords(
            torch.zeros((1, 2, 4, 4, 4)), patches, torch.tensor([[0, 0, 0]])
        )
        self.assertEqual(float(output[:, :, 3, 3, 3].sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
