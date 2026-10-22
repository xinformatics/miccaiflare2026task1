import unittest

from select_focus_epoch_routes import decide


def baseline(case=0.50, cohort=0.48, negative=0.90):
    return {
        "case_balanced_segmentation_dice": case,
        "cohort_balanced_segmentation_dice": cohort,
        "negative_nonempty_rate": negative,
    }


def candidate(case=0.51, cohort=0.49, negative=0.91, folds=5):
    return {
        "folds": folds,
        "fold_records": [{} for _ in range(folds)],
        "route_groups": ["mature", "weak_sources", "lung_cells"],
        "segmentation": {
            "dice": case,
            "cohort_balanced_dice": cohort,
            "negative_nonempty_rate": negative,
        },
    }


class FocusEpochRouteSelectionTests(unittest.TestCase):
    def test_meaningful_bounded_gain_qualifies(self):
        decision, metrics, checks = decide(baseline(), candidate())
        self.assertEqual(decision, "QUALIFY_FOCUS_EPOCH_ROUTES_FULL_VOLUME")
        self.assertTrue(all(checks.values()))
        self.assertAlmostEqual(metrics["case_dice_gain"], 0.01)

    def test_small_gain_or_negative_increase_fails(self):
        decision, _, checks = decide(baseline(), candidate(case=0.503))
        self.assertEqual(decision, "KEEP_GLOBAL_FOCUS_EPOCHS")
        self.assertFalse(checks["meaningful_case_gain"])
        decision, _, checks = decide(baseline(), candidate(negative=0.93))
        self.assertEqual(decision, "KEEP_GLOBAL_FOCUS_EPOCHS")
        self.assertFalse(checks["negative_rate_bounded"])

    def test_incomplete_routes_fail(self):
        value = candidate(folds=4)
        value["route_groups"].pop()
        decision, _, checks = decide(baseline(), value)
        self.assertEqual(decision, "KEEP_GLOBAL_FOCUS_EPOCHS")
        self.assertFalse(checks["route_groups_complete"])
        self.assertFalse(checks["fivefold_complete"])


if __name__ == "__main__":
    unittest.main()
