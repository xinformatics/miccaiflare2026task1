import unittest

from select_candidate_refiner_decoupled_oof import decide


def summary(ap=0.80, precision=0.75, recall=0.60, dice=0.55, prevalence=0.30):
    return {
        "classification": {
            "average_precision": ap,
            "precision": precision,
            "recall": recall,
            "prevalence": prevalence,
        },
        "segmentation": {"dice": dice},
    }


class DecoupledOOFSelectionTests(unittest.TestCase):
    def test_ap_gain_with_preserved_contour_qualifies_full_volume_only(self):
        decision, gains, checks = decide(
            summary(), summary(ap=0.805, dice=0.55)
        )
        self.assertEqual(decision, "QUALIFY_DECOUPLED_FULL_VOLUME")
        self.assertTrue(all(checks.values()))
        self.assertAlmostEqual(gains["average_precision"], 0.005)

    def test_any_primary_regression_or_tiny_gain_keeps_coupled(self):
        decision, _, checks = decide(
            summary(), summary(ap=0.81, dice=0.549)
        )
        self.assertEqual(decision, "KEEP_COUPLED")
        self.assertFalse(checks["segmentation_nonregression"])
        decision, _, checks = decide(
            summary(), summary(ap=0.801, dice=0.55)
        )
        self.assertEqual(decision, "KEEP_COUPLED")
        self.assertFalse(checks["meaningful_primary_gain"])

    def test_ineligible_classifier_fails_closed(self):
        decision, _, checks = decide(
            summary(), summary(ap=0.43, precision=0.49, dice=0.56)
        )
        self.assertEqual(decision, "KEEP_COUPLED")
        self.assertFalse(checks["candidate_classifier_eligible"])


if __name__ == "__main__":
    unittest.main()
