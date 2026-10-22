import unittest
from types import SimpleNamespace

import torch

from lesionlocator.inference.lesionlocator_segment import LesionLocatorSegmenter


class _ConstantForeground(torch.nn.Module):
    def forward(self, value):
        batch, _, depth, height, width = value.shape
        logits = torch.zeros((batch, 2, depth, height, width), dtype=value.dtype)
        logits[:, 1] = 1
        return logits


class BoxFocusCoverageTests(unittest.TestCase):
    def test_unvisited_voxels_have_finite_background_tie_logits(self):
        predictor = LesionLocatorSegmenter.__new__(LesionLocatorSegmenter)
        predictor.device = torch.device("cpu")
        predictor.network = _ConstantForeground()
        predictor.label_manager = SimpleNamespace(num_segmentation_heads=2)
        predictor.use_gaussian = False
        predictor.use_mirroring = False
        predictor.allow_tqdm = False
        predictor.verbose = False

        data = torch.zeros((2, 4, 4, 4), dtype=torch.float32)
        covered = (
            slice(None),
            slice(0, 2),
            slice(0, 2),
            slice(0, 2),
        )
        logits = predictor._internal_predict_sliding_window_return_logits(
            data, [covered], do_on_device=False
        )

        self.assertTrue(torch.isfinite(logits).all())
        torch.testing.assert_close(
            logits[:, 3, 3, 3], torch.zeros_like(logits[:, 3, 3, 3])
        )
        self.assertEqual(float(logits[1, 0, 0, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()
