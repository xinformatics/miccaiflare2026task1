import unittest

from summarize_focus_routed_tta import qualification


class SummarizeFocusRoutedTTATests(unittest.TestCase):
    def test_all_frozen_gates_must_pass(self):
        passed = qualification(
            {
                "case_balanced_dice": 0.003,
                "cohort_balanced_dice": 0.003,
                "selection_score": 0.002,
                "negative_nonempty_rate": 0.01,
            }
        )
        self.assertTrue(all(passed.values()))
        failed = qualification(
            {
                "case_balanced_dice": 0.003,
                "cohort_balanced_dice": 0.001,
                "selection_score": 0.002,
                "negative_nonempty_rate": 0.01,
            }
        )
        self.assertFalse(failed["cohort_gain"])
        self.assertFalse(all(failed.values()))

    def test_negative_activation_increase_is_bounded(self):
        checks = qualification(
            {
                "case_balanced_dice": 0.003,
                "cohort_balanced_dice": 0.003,
                "selection_score": 0.002,
                "negative_nonempty_rate": 0.021,
            }
        )
        self.assertFalse(checks["negative_increase"])


if __name__ == "__main__":
    unittest.main()
