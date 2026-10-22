import unittest
from pathlib import Path

from audit_center_small_lesion_recall import (
    deployed_from_metrics,
    raw_from_metrics,
    small_lesion_decision,
)


def summary(detected_100to500, detected_500to1000, precision):
    return {
        "gt_by_size_mm3": {
            "100to500": {"detected": detected_100to500},
            "500to1000": {"detected": detected_500to1000},
        },
        "components": {"precision": precision},
    }


class CenterSmallLesionAuditTests(unittest.TestCase):
    def test_deployed_path_is_derived_from_metrics_contract(self):
        metrics = Path("root/arm/checkpoint/g0p20.metrics.csv")
        self.assertEqual(
            deployed_from_metrics(metrics),
            Path("root/arm/checkpoint/g0p20/deployed_mm1000_new2k_s25"),
        )
        self.assertEqual(
            raw_from_metrics(metrics),
            Path("root/arm/checkpoint/g0p20/inference"),
        )

    def test_one_extra_detection_without_regression_qualifies(self):
        decision = small_lesion_decision(
            summary(2, 7, 0.59), summary(3, 7, 0.57)
        )
        self.assertTrue(decision["qualified"])
        self.assertEqual(decision["small_detected_gain"], 1)

    def test_gain_cannot_hide_a_bin_regression(self):
        decision = small_lesion_decision(
            summary(2, 7, 0.59), summary(4, 6, 0.59)
        )
        self.assertFalse(decision["qualified"])
        self.assertFalse(decision["no_bin_regression"])

    def test_precision_collapse_is_rejected(self):
        decision = small_lesion_decision(
            summary(2, 7, 0.59), summary(3, 7, 0.559)
        )
        self.assertFalse(decision["qualified"])
        self.assertFalse(decision["component_precision_guard"])


if __name__ == "__main__":
    unittest.main()
