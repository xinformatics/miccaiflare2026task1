import copy
import unittest

from select_candidate_refiner_union_native_pilot import select


def candidate(family, dice, score, ap):
    return {
        "family": family,
        "checkpoint": f"/{family}.pt",
        "metrics": {
            "case_balanced_segmentation_dice": dice,
            "selection_score": score,
            "average_precision": ap,
        },
    }


class UnionNativePilotSelectorTests(unittest.TestCase):
    def setUp(self):
        self.plan = {"fold0_gate_all_required": {
            "case_balanced_segmentation_dice_gain_min": 0.015,
            "case_balanced_selection_score_gain_min": 0.01,
            "average_precision_regression_max": 0.005,
        }}
        self.baseline = {
            "selected_family": "composite",
            "selected_checkpoint": "/transfer.pt",
            "selected_metrics": {
                "case_balanced_segmentation_dice": 0.50,
                "selection_score": 0.61,
                "average_precision": 0.72,
            },
        }

    def test_all_gates_are_required(self):
        candidates = [
            candidate("composite", 0.520, 0.625, 0.716),
            candidate("segmentation", 0.530, 0.615, 0.74),
            candidate("classification", 0.51, 0.63, 0.75),
        ]
        result = select(self.plan, self.baseline, candidates)
        self.assertTrue(result["passed"])
        self.assertEqual(result["selected_family"], "composite")

    def test_dice_only_gain_does_not_pass(self):
        candidates = [
            candidate("composite", 0.530, 0.615, 0.74),
            candidate("segmentation", 0.52, 0.619, 0.72),
            candidate("classification", 0.50, 0.61, 0.72),
        ]
        result = select(self.plan, self.baseline, candidates)
        self.assertFalse(result["passed"])
        self.assertEqual(result["decision"], "KEEP_TRANSFER")

    def test_missing_family_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "all three"):
            select(
                self.plan,
                self.baseline,
                [candidate("composite", 0.53, 0.63, 0.73)],
            )


if __name__ == "__main__":
    unittest.main()
