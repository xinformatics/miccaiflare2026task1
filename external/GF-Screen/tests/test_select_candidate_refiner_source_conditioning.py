import unittest

from select_candidate_refiner_source_conditioning import decide, summarize


def row(epoch, segmentation, average_precision, selection):
    return {
        "epoch": epoch,
        "segmentation_dice": segmentation,
        "average_precision": average_precision,
        "selection_score": selection,
    }


class SourceConditioningSelectionTests(unittest.TestCase):
    def test_summary_selects_each_checkpoint_family_independently(self):
        result = summarize(
            [
                row(1, 0.50, 0.90, 0.70),
                row(2, 0.55, 0.80, 0.68),
                row(3, 0.52, 0.88, 0.72),
            ]
        )
        self.assertEqual(result["segmentation"]["epoch"], 2)
        self.assertEqual(result["classification"]["epoch"], 1)
        self.assertEqual(result["composite"]["epoch"], 3)

    def test_all_three_predeclared_gates_are_required(self):
        baseline = summarize([row(1, 0.50, 0.80, 0.65)])
        promoted = summarize([row(1, 0.53, 0.795, 0.66)])
        decision, gains, gates = decide(baseline, promoted)
        self.assertEqual(decision, "EXPAND_SOURCE_CONDITIONING")
        self.assertTrue(all(gates.values()))
        self.assertAlmostEqual(gains["segmentation_dice"], 0.03)

        classification_failure = summarize([row(1, 0.53, 0.78, 0.66)])
        decision, _, gates = decide(baseline, classification_failure)
        self.assertEqual(decision, "KEEP_UNCONDITIONED")
        self.assertFalse(gates["classification_preserved"])

        segmentation_failure = summarize([row(1, 0.519, 0.82, 0.67)])
        decision, _, gates = decide(baseline, segmentation_failure)
        self.assertEqual(decision, "KEEP_UNCONDITIONED")
        self.assertFalse(gates["segmentation_gain"])

    def test_ties_prefer_earlier_epoch_after_metric_tie_breaks(self):
        result = summarize(
            [row(2, 0.5, 0.8, 0.65), row(1, 0.5, 0.8, 0.65)]
        )
        self.assertEqual(result["segmentation"]["epoch"], 1)
        self.assertEqual(result["classification"]["epoch"], 1)
        self.assertEqual(result["composite"]["epoch"], 1)

    def test_invalid_history_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            summarize([])
        with self.assertRaisesRegex(ValueError, "misses"):
            summarize([{"epoch": 1}])


if __name__ == "__main__":
    unittest.main()
