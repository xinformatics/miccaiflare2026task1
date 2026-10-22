import unittest

import numpy as np
import torch

from docker_submission.shared_focus_dual_glance import (
    crop_coordinates,
    dual_glance_shared_focus_predict,
)


class ThresholdGlance(torch.nn.Module):
    def __init__(self, *, lower=None, upper=None):
        super().__init__()
        self.lower = lower
        self.upper = upper

    def get_actions(self, patches):
        means = patches.mean(dim=(1, 2, 3, 4))
        keep = torch.ones_like(means, dtype=torch.bool)
        if self.lower is not None:
            keep &= means > self.lower
        if self.upper is not None:
            keep &= means < self.upper
        score = torch.where(keep, torch.full_like(means, 8.0), torch.full_like(means, -8.0))
        return torch.stack((-score, score), dim=1)


class CountingFocus(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patches_seen = 0

    def forward(self, patches):
        self.patches_seen += int(patches.shape[0])
        foreground = patches[:, :1] - 2.5
        return torch.cat((-foreground, foreground), dim=1)


class SharedFocusDualGlanceTests(unittest.TestCase):
    def setUp(self):
        x = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1, 1)
        self.volume = x.expand(1, 8, 4, 4).contiguous()
        self.kwargs = dict(
            crop=(4, 4, 4), overlaps=(0.5, 0.0, 0.0),
            glance_batch_size=2, focus_batch_size=2,
            device="cpu", autocast=False,
        )

    def test_coordinate_grid_has_no_duplicate_tail_windows(self):
        self.assertEqual(
            crop_coordinates((8, 4, 4), (4, 4, 4), (0.5, 0.0, 0.0)),
            [(0, 0, 0), (2, 0, 0), (4, 0, 0)],
        )

    def test_shared_result_matches_two_independent_routes(self):
        route_a = ThresholdGlance(upper=4.0)
        route_b = ThresholdGlance(lower=2.0)
        shared_focus = CountingFocus()
        shared_a, shared_b, stats = dual_glance_shared_focus_predict(
            route_a, route_b, shared_focus, self.volume,
            glance_threshold_a=0.5, glance_threshold_b=0.5, **self.kwargs,
        )
        focus_a = CountingFocus()
        independent_a, _, _ = dual_glance_shared_focus_predict(
            route_a, route_a, focus_a, self.volume,
            glance_threshold_a=0.5, glance_threshold_b=0.5, **self.kwargs,
        )
        focus_b = CountingFocus()
        independent_b, _, _ = dual_glance_shared_focus_predict(
            route_b, route_b, focus_b, self.volume,
            glance_threshold_a=0.5, glance_threshold_b=0.5, **self.kwargs,
        )
        np.testing.assert_array_equal(shared_a, independent_a)
        np.testing.assert_array_equal(shared_b, independent_b)
        self.assertEqual(stats.grid_crops, 3)
        self.assertEqual(stats.route_a_crops, 2)
        self.assertEqual(stats.route_b_crops, 2)
        self.assertEqual(stats.union_crops, 3)
        self.assertEqual(stats.reused_crops, 1)
        self.assertEqual(shared_focus.patches_seen, 3)
        self.assertEqual(focus_a.patches_seen + focus_b.patches_seen, 4)

    def test_rejects_invalid_runtime_contracts(self):
        focus = CountingFocus()
        glance = ThresholdGlance()
        with self.assertRaisesRegex(ValueError, "glance_threshold_a"):
            dual_glance_shared_focus_predict(
                glance, glance, focus, self.volume,
                glance_threshold_a=1.1, **self.kwargs,
            )
        with self.assertRaisesRegex(ValueError, "batch sizes"):
            dual_glance_shared_focus_predict(
                glance, glance, focus, self.volume,
                focus_batch_size=0,
                crop=(4, 4, 4), overlaps=(0.5, 0.0, 0.0),
                device="cpu", autocast=False,
            )


if __name__ == "__main__":
    unittest.main()
