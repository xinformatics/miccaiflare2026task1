import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT.parents[1] / "handoff_bundle" / "code"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OFFICIAL))

from evaluate import score_case as official_score_case
from fast_official_lesion_metrics import (
    dense_surface_dice_at_tolerance,
    prepare_surface_reference,
    score_case as fast_score_case,
    score_case_with_reference,
    surface_dice_at_tolerance,
)


class FastOfficialLesionMetricTests(unittest.TestCase):
    def assert_metric_equal(self, ground_truth, prediction, spacing):
        expected = official_score_case(ground_truth, prediction, spacing)
        observed = fast_score_case(ground_truth, prediction, spacing)
        np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-12)

    def test_decision_ladder(self):
        empty = np.zeros((9, 10, 11), dtype=np.uint8)
        one = empty.copy(); one[4, 4, 4] = 1
        invalid = one.copy(); invalid[4, 4, 4] = 2
        for ground_truth, prediction in ((empty, empty), (empty, one), (one, empty), (one, invalid)):
            self.assert_metric_equal(ground_truth, prediction, (1.0, 1.0, 1.0))

    def test_random_masks_and_anisotropic_spacings(self):
        rng = np.random.default_rng(20260824)
        for spacing in ((1.0, 1.0, 1.0), (0.7, 0.8, 2.5), (1.5, 1.5, 3.0)):
            for _ in range(24):
                ground_truth = rng.binomial(1, 0.08, size=(13, 15, 17)).astype(np.uint8)
                prediction = ground_truth.copy()
                prediction ^= rng.binomial(1, 0.03, size=prediction.shape).astype(np.uint8)
                self.assert_metric_equal(ground_truth, prediction, spacing)
                reference = prepare_surface_reference(ground_truth, spacing)
                np.testing.assert_allclose(
                    score_case_with_reference(
                        ground_truth, prediction, spacing, reference
                    ),
                    official_score_case(ground_truth, prediction, spacing),
                    rtol=0,
                    atol=1e-12,
                )
                self.assertAlmostEqual(
                    surface_dice_at_tolerance(ground_truth, prediction, spacing),
                    dense_surface_dice_at_tolerance(ground_truth, prediction, spacing),
                    places=12,
                )

    def test_disconnected_and_boundary_touching_components(self):
        ground_truth = np.zeros((20, 22, 24), dtype=np.uint8)
        prediction = np.zeros_like(ground_truth)
        ground_truth[0:4, 0:5, 0:6] = 1
        ground_truth[14:18, 16:20, 19:24] = 1
        prediction[1:5, 0:5, 1:7] = 1
        prediction[12:16, 15:19, 18:23] = 1
        self.assert_metric_equal(ground_truth, prediction, (0.8, 0.9, 2.0))


if __name__ == "__main__":
    unittest.main()
