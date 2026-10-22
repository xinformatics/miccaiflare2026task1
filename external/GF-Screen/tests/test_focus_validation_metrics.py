import unittest

import numpy as np

from focus_validation_metrics import (
    aggregate_positive_dice,
    apply_case_balanced_metrics,
)


class FocusValidationMetricsTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"patch_id": "a0", "case": "case_a", "cohort": "x", "label": "1"},
            {"patch_id": "a1", "case": "case_a", "cohort": "x", "label": "1"},
            {"patch_id": "a2", "case": "case_a", "cohort": "x", "label": "1"},
            {"patch_id": "b0", "case": "case_b", "cohort": "y", "label": "1"},
        ]
        self.ids = np.asarray(["a0", "a1", "a2", "b0"])
        self.grid = np.asarray(
            [[1.0, 0.4], [1.0, 0.4], [1.0, 0.4], [0.0, 1.0]],
            dtype=np.float32,
        )
        self.thresholds = np.asarray([0.4, 0.6], dtype=np.float32)

    def test_case_balance_prevents_many_tiles_from_dominating(self):
        result = aggregate_positive_dice(
            self.ids, self.grid, self.thresholds, self.rows
        )
        self.assertAlmostEqual(result["threshold"], 0.6, places=5)
        self.assertAlmostEqual(result["case_balanced_dice"], 0.7, places=5)
        self.assertAlmostEqual(result["patch_mean_dice"], 0.55, places=5)
        self.assertEqual(result["positive_cases"], 2)
        self.assertEqual(result["positive_cohorts"], 2)

    def test_apply_uses_threshold_matched_negative_rate(self):
        metrics = {"negative_patches": 10, "negative_nonempty_rate": 1.0}
        predictions = {
            "positive_patch_id": self.ids,
            "segmentation_dice_grid": self.grid,
            "segmentation_thresholds": self.thresholds,
            "negative_nonempty_by_threshold": np.asarray([10, 2]),
        }
        result = apply_case_balanced_metrics(metrics, predictions, self.rows)
        self.assertAlmostEqual(result["segmentation_threshold"], 0.6, places=5)
        self.assertAlmostEqual(result["negative_nonempty_rate"], 0.2, places=5)
        self.assertAlmostEqual(result["selection_score"], 0.696, places=5)

    def test_duplicate_positive_id_fails(self):
        with self.assertRaises(ValueError):
            aggregate_positive_dice(
                ["a0", "a0"], self.grid[:2], self.thresholds, self.rows
            )

    def test_incomplete_positive_coverage_fails(self):
        with self.assertRaises(ValueError):
            aggregate_positive_dice(
                self.ids[:3], self.grid[:3], self.thresholds, self.rows
            )


if __name__ == "__main__":
    unittest.main()
