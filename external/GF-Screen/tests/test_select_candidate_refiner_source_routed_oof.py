import unittest

from select_candidate_refiner_source_routed_oof import decide


def summary(ap=0.8, dice=0.5, mature=0.6, weak=0.4, cells=0.3):
    return {
        "classification": {"average_precision": ap},
        "segmentation": {"dice": dice},
        "source_segmentation_groups": {
            "mature": {"dice": mature},
            "weak_sources": {"dice": weak},
            "lung_cells": {"dice": cells},
        },
    }


class SourceRoutedSelectionTests(unittest.TestCase):
    def test_global_gain_with_group_protection_qualifies(self):
        decision, metrics, checks = decide(
            summary(), summary(dice=0.505, mature=0.599, weak=0.405, cells=0.31)
        )
        self.assertEqual(decision, "QUALIFY_SOURCE_ROUTED_FULL_VOLUME")
        self.assertTrue(all(checks.values()))
        self.assertAlmostEqual(metrics["segmentation_dice_gain"], 0.005)

    def test_classification_change_fails_closed(self):
        decision, _, checks = decide(summary(), summary(ap=0.8001, dice=0.51))
        self.assertEqual(decision, "KEEP_GLOBAL_CONTOUR")
        self.assertFalse(checks["classification_identical"])

    def test_group_regression_or_missing_group_fails_closed(self):
        decision, _, checks = decide(
            summary(), summary(dice=0.51, mature=0.597, weak=0.41, cells=0.31)
        )
        self.assertEqual(decision, "KEEP_GLOBAL_CONTOUR")
        self.assertFalse(checks["source_group_nonregression"])
        candidate = summary(dice=0.51, weak=0.41, cells=0.31)
        del candidate["source_segmentation_groups"]["mature"]
        decision, _, checks = decide(summary(), candidate)
        self.assertEqual(decision, "KEEP_GLOBAL_CONTOUR")
        self.assertFalse(checks["required_source_groups_present"])


if __name__ == "__main__":
    unittest.main()
