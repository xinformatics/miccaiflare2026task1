import unittest

import torch

from evaluate_focus_checkpoint_tta import (
    FocusTTAProbabilityAsLogits,
    parse_tta_axes,
)
from focus_candidate_refiner import foreground_logit


class FocusCheckpointTTATests(unittest.TestCase):
    def test_axis_parser_matches_champion_coordinate_convention(self):
        self.assertEqual(parse_tta_axes(""), ())
        self.assertEqual(parse_tta_axes("x,y"), (2, 3))
        self.assertEqual(parse_tta_axes(" z "), (4,))
        with self.assertRaisesRegex(ValueError, "only x, y, and z"):
            parse_tta_axes("q")
        with self.assertRaisesRegex(ValueError, "repeat"):
            parse_tta_axes("xx")

    def test_empty_tta_is_probability_equivalent_to_base_model(self):
        class Base(torch.nn.Module):
            def forward(self, images):
                return torch.cat((-images[:, :1], images[:, :1]), dim=1)

        images = torch.randn(2, 1, 3, 4, 5)
        base = Base()
        expected = torch.sigmoid(foreground_logit(base(images)))
        wrapped = FocusTTAProbabilityAsLogits(base, ())
        observed = torch.sigmoid(foreground_logit(wrapped(images)))
        torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)

    def test_flip_prediction_is_restored_before_averaging(self):
        class Directional(torch.nn.Module):
            def forward(self, images):
                # A fixed coordinate bias makes the model intentionally
                # non-equivariant, so the expected TTA average is observable.
                coordinate = torch.arange(
                    images.shape[2], dtype=images.dtype, device=images.device
                ).view(1, 1, -1, 1, 1)
                foreground = images[:, :1] + coordinate
                return torch.cat((torch.zeros_like(foreground), foreground), dim=1)

        images = torch.randn(1, 1, 3, 4, 5)
        base = Directional()
        original = torch.sigmoid(foreground_logit(base(images)))
        flipped = torch.flip(images, dims=(2,))
        restored = torch.flip(
            torch.sigmoid(foreground_logit(base(flipped))), dims=(2,)
        )
        expected = (original + restored) / 2
        wrapped = FocusTTAProbabilityAsLogits(base, (2,))
        observed = torch.sigmoid(foreground_logit(wrapped(images)))
        torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
