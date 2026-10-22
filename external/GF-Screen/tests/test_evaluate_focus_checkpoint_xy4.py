import unittest

import torch

from evaluate_focus_checkpoint_xy4 import FLIP_SETS, FocusXY4ProbabilityAsLogits
from focus_candidate_refiner import foreground_logit


class FocusCheckpointXY4Tests(unittest.TestCase):
    def test_uses_complete_xy_flip_group(self):
        self.assertEqual(FLIP_SETS, ((), (2,), (3,), (2, 3)))

    def test_each_prediction_is_restored_before_four_view_average(self):
        class Directional(torch.nn.Module):
            def forward(self, images):
                x = torch.arange(
                    images.shape[2], dtype=images.dtype, device=images.device
                ).view(1, 1, -1, 1, 1)
                y = torch.arange(
                    images.shape[3], dtype=images.dtype, device=images.device
                ).view(1, 1, 1, -1, 1)
                foreground = images[:, :1] + x + 2 * y
                return torch.cat((torch.zeros_like(foreground), foreground), dim=1)

        images = torch.randn(2, 1, 3, 4, 5)
        base = Directional()
        restored = []
        for dimensions in FLIP_SETS:
            view = torch.flip(images, dims=dimensions) if dimensions else images
            probability = torch.sigmoid(foreground_logit(base(view)))
            if dimensions:
                probability = torch.flip(probability, dims=dimensions)
            restored.append(probability)
        expected = torch.stack(restored).mean(dim=0)
        observed = torch.sigmoid(
            foreground_logit(FocusXY4ProbabilityAsLogits(base)(images))
        )
        torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
