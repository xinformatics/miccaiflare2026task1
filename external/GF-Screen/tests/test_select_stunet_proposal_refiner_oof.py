import unittest

import numpy as np

from select_stunet_proposal_refiner_oof import (
    PRECISION_MINIMUM,
    THRESHOLDS,
    bootstrap_precision_interval,
    choose_threshold,
)


class StunetProposalRefinerGateTests(unittest.TestCase):
    def test_threshold_grid_is_predeclared_and_bounded(self):
        self.assertEqual(THRESHOLDS, (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95))
        self.assertEqual(PRECISION_MINIMUM, 0.90)

    def test_maximizes_recall_subject_to_precision_floor(self):
        labels = np.asarray([1, 1, 1, 0, 0, 0], dtype=np.uint8)
        probabilities = np.asarray([0.96, 0.91, 0.84, 0.83, 0.20, 0.10])
        cases = np.asarray([f"case{i}" for i in range(6)])
        selected, rows = choose_threshold(labels, probabilities, cases)
        self.assertEqual(selected["threshold"], 0.85)
        self.assertEqual(selected["precision"], 1.0)
        self.assertAlmostEqual(selected["recall"], 2 / 3)
        self.assertEqual(len(rows), 10)

    def test_no_qualifying_threshold_retains_precision_diagnostic(self):
        labels = np.asarray([1, 0, 0], dtype=np.uint8)
        probabilities = np.asarray([0.80, 0.99, 0.98])
        cases = np.asarray(["a", "b", "c"])
        selected, _ = choose_threshold(labels, probabilities, cases)
        self.assertLess(selected["precision"], PRECISION_MINIMUM)

    def test_case_bootstrap_precision_is_finite(self):
        labels = np.asarray([1, 0, 1, 0, 1, 0], dtype=np.uint8)
        probabilities = np.asarray([0.99, 0.1, 0.95, 0.2, 0.9, 0.3])
        cases = np.asarray(["a", "a", "b", "b", "c", "c"])
        interval = bootstrap_precision_interval(labels, probabilities, cases, 0.8, 1000, 7)
        self.assertEqual(interval, [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
