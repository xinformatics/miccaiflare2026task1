import unittest

from select_candidate_refiner_union_native_confirmation import select


def candidate(family, dice_gain, selection_gain, ap_gain):
    baseline = {"dice": 0.50, "selection": 0.60, "ap": 0.70}
    return {
        "family": family,
        "checkpoint": f"/{family}.pt",
        "metrics": {
            "case_balanced_segmentation_dice": baseline["dice"] + dice_gain,
            "selection_score": baseline["selection"] + selection_gain,
            "average_precision": baseline["ap"] + ap_gain,
        },
    }


class UnionNativeConfirmationSelectorTests(unittest.TestCase):
    def setUp(self):
        self.plan = {"future_untouched_fold1_gate_if_fold0_passes": {
            "fold1_case_balanced_segmentation_dice_gain_min": 0.005,
            "fold1_selection_score_gain_min": 0.0,
            "two_fold_mean_case_balanced_segmentation_dice_gain_min": 0.015,
            "two_fold_mean_selection_score_gain_min": 0.01,
            "average_precision_regression_max_each_fold": 0.01,
        }}
        self.fold0 = {
            "decision": "CONFIRM_ON_UNTOUCHED_FOLD", "passed": True,
            "selected_family": "composite", "selected_checkpoint": "/fold0.pt",
            "selected_gains": {
                "case_balanced_segmentation_dice": 0.020,
                "selection_score": 0.015,
                "average_precision": 0.010,
            },
        }
        self.baseline = {
            "fold": 1, "selected_family": "segmentation",
            "selected_checkpoint": "/transfer1.pt",
            "selected_metrics": {
                "case_balanced_segmentation_dice": 0.50,
                "selection_score": 0.60,
                "average_precision": 0.70,
            },
        }

    def test_requires_reproduced_untouched_gain(self):
        candidates = [
            candidate("composite", 0.012, 0.006, -0.005),
            candidate("segmentation", 0.004, 0.020, 0.02),
            candidate("classification", 0.020, -0.001, 0.03),
        ]
        result = select(self.plan, self.fold0, self.baseline, candidates)
        self.assertTrue(result["passed"])
        self.assertEqual(result["decision"], "EXPAND_FIVE_FOLD")
        self.assertEqual(result["fold1_selected_family"], "composite")

    def test_fold0_cannot_carry_a_fold1_regression(self):
        candidates = [
            candidate("composite", -0.005, 0.020, 0.02),
            candidate("segmentation", 0.020, -0.001, 0.02),
            candidate("classification", 0.010, 0.010, -0.02),
        ]
        result = select(self.plan, self.fold0, self.baseline, candidates)
        self.assertFalse(result["passed"])
        self.assertEqual(result["decision"], "KEEP_TRANSFER")

    def test_rejects_unapproved_fold0(self):
        fold0 = dict(self.fold0, passed=False)
        with self.assertRaisesRegex(ValueError, "did not pass"):
            select(
                self.plan, fold0, self.baseline,
                [candidate(name, 0.02, 0.02, 0) for name in ("composite", "segmentation", "classification")],
            )


if __name__ == "__main__":
    unittest.main()
