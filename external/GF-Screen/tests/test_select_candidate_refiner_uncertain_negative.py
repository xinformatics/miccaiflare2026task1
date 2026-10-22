import unittest

from select_candidate_refiner_uncertain_negative import decide


def summary(segmentation: float, composite: float, classification: float):
    return {
        "segmentation": {"segmentation_dice": segmentation},
        "composite": {"selection_score": composite},
        "classification": {"average_precision": classification},
    }


class UncertainNegativeSelectorTests(unittest.TestCase):
    def test_requires_all_three_matched_fold_gates(self):
        baseline = summary(0.60, 0.70, 0.80)
        decision, gains, gates = decide(
            baseline, summary(0.596, 0.703, 0.806)
        )
        self.assertEqual(decision, "EXPAND_UNCERTAIN_NEGATIVE_025")
        self.assertTrue(all(gates.values()))
        self.assertAlmostEqual(gains["average_precision"], 0.006)

        for candidate in (
            summary(0.594, 0.703, 0.806),
            summary(0.596, 0.701, 0.806),
            summary(0.596, 0.703, 0.804),
        ):
            decision, _, gates = decide(baseline, candidate)
            self.assertEqual(decision, "KEEP_HARD_NEGATIVES")
            self.assertFalse(all(gates.values()))


if __name__ == "__main__":
    unittest.main()
