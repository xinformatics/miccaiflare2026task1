import sys
import unittest
from pathlib import Path

import numpy as np


OFFICIAL_CODE = Path(__file__).resolve().parents[3] / "handoff_bundle" / "code"
sys.path.insert(0, str(OFFICIAL_CODE))
from evaluate import score_case  # noqa: E402


class OfficialMetricDecisionLadderTests(unittest.TestCase):
    def setUp(self):
        self.spacing = (1.0, 1.0, 1.0)

    def test_empty_reference_and_prediction_score_one(self):
        empty = np.zeros((8, 8, 8), dtype=np.uint8)
        self.assertEqual(score_case(empty, empty, self.spacing), (1.0, 1.0))

    def test_false_positive_on_empty_reference_scores_zero(self):
        ground_truth = np.zeros((8, 8, 8), dtype=np.uint8)
        prediction = ground_truth.copy()
        prediction[3, 3, 3] = 1
        self.assertEqual(score_case(ground_truth, prediction, self.spacing), (0.0, 0.0))

    def test_empty_prediction_on_lesion_case_scores_zero(self):
        ground_truth = np.zeros((8, 8, 8), dtype=np.uint8)
        ground_truth[3, 3, 3] = 1
        prediction = np.zeros_like(ground_truth)
        self.assertEqual(score_case(ground_truth, prediction, self.spacing), (0.0, 0.0))

    def test_nonbinary_prediction_scores_zero(self):
        ground_truth = np.zeros((8, 8, 8), dtype=np.uint8)
        prediction = ground_truth.copy()
        ground_truth[3, 3, 3] = 1
        prediction[3, 3, 3] = 2
        self.assertEqual(score_case(ground_truth, prediction, self.spacing), (0.0, 0.0))

    def test_dsc_below_point_two_forces_nsd_to_zero(self):
        ground_truth = np.zeros((12, 8, 8), dtype=np.uint8)
        ground_truth[1:11, 3, 3] = 1
        prediction = np.zeros_like(ground_truth)
        prediction[1, 3, 3] = 1
        dsc, nsd = score_case(ground_truth, prediction, self.spacing)
        self.assertLess(dsc, 0.2)
        self.assertEqual(nsd, 0.0)

    def test_identical_nonempty_masks_score_one(self):
        ground_truth = np.zeros((8, 8, 8), dtype=np.uint8)
        ground_truth[2:6, 2:6, 2:6] = 1
        dsc, nsd = score_case(ground_truth, ground_truth.copy(), self.spacing)
        self.assertAlmostEqual(dsc, 1.0)
        self.assertAlmostEqual(nsd, 1.0)


if __name__ == "__main__":
    unittest.main()
