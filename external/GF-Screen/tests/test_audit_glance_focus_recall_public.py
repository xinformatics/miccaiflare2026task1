from __future__ import annotations

import unittest

import numpy as np

from audit_glance_focus_recall_public import dice, overlap_fraction, score_volume, size_bin


class GlanceFocusRecallAuditTests(unittest.TestCase):
    def test_score_volume_is_maximum_probability_of_covering_crops(self):
        score = score_volume(
            (4, 4, 4),
            [(0, 0, 0), (2, 2, 2)],
            np.asarray([0.3, 0.8]),
            crop_shape=(2, 2, 2),
        )
        self.assertAlmostEqual(float(score[0, 0, 0]), 0.3)
        self.assertAlmostEqual(float(score[2, 2, 2]), 0.8)
        self.assertEqual(float(score[1, 2, 2]), 0.0)

    def test_overlap_fraction_and_dice(self):
        component = np.zeros((2, 2, 2), dtype=bool)
        component[0] = True
        prediction = np.zeros_like(component)
        prediction[0, 0] = True
        self.assertEqual(overlap_fraction(component, prediction), 0.5)
        self.assertEqual(dice(component, prediction), 2.0 / 3.0)

    def test_physical_size_bands_have_exact_boundaries(self):
        self.assertEqual(size_bin(99.99), "lt100")
        self.assertEqual(size_bin(100.0), "100to500")
        self.assertEqual(size_bin(500.0), "500to1000")
        self.assertEqual(size_bin(1000.0), "1000to3000")
        self.assertEqual(size_bin(3000.0), "ge3000")


if __name__ == "__main__":
    unittest.main()
