import unittest

from select_candidate_refiner_variant_oof import decide


def summary(ap=0.80, dice=0.55, precision=0.8, recall=0.7, prevalence=0.3):
    return {
        "classification": {
            "average_precision": ap,
            "precision": precision,
            "recall": recall,
            "prevalence": prevalence,
        },
        "segmentation": {"dice": dice},
    }


class CandidateRefinerVariantOOFTests(unittest.TestCase):
    def test_qualifies_only_meaningful_segmentation_and_composite_gain(self):
        decision, gains, checks = decide(
            summary(), summary(ap=0.795, dice=0.565)
        )
        self.assertEqual(decision, "QUALIFY_VARIANT_FULL_VOLUME")
        self.assertAlmostEqual(gains["segmentation_dice"], 0.015)
        self.assertTrue(all(checks.values()))

    def test_rejects_ap_drop_or_ineligible_classifier(self):
        self.assertEqual(
            decide(summary(), summary(ap=0.78, dice=0.58))[0], "KEEP_PRIMARY"
        )
        self.assertEqual(
            decide(summary(), summary(ap=0.81, dice=0.56, precision=0.4))[0],
            "KEEP_PRIMARY",
        )

    def test_rejects_dice_gain_that_is_too_small(self):
        self.assertEqual(
            decide(summary(), summary(ap=0.81, dice=0.553))[0], "KEEP_PRIMARY"
        )


if __name__ == "__main__":
    unittest.main()
