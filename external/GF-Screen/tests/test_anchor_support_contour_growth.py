import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from screen_anchor_support_contour_growth import transform  # noqa: E402


class AnchorSupportContourGrowthTests(unittest.TestCase):
    def test_finite_growth_preserves_anchor_and_stays_in_support(self):
        anchor = np.zeros((11, 11, 11), dtype=bool)
        anchor[5, 5, 5] = True
        support = np.zeros_like(anchor)
        support[5, 5, 5:10] = True
        result = transform(anchor, support, (1.0, 1.0, 1.0), "support_2mm")
        self.assertTrue(result[5, 5, 5:8].all())
        self.assertFalse(result[5, 5, 8])
        self.assertTrue(np.all(result[anchor]))
        self.assertFalse(np.any((result & ~anchor) & ~support))


    def test_touching_component_does_not_import_disconnected_support(self):
        anchor = np.zeros((12, 12, 12), dtype=bool)
        anchor[2:4, 2:4, 2:4] = True
        support = np.zeros_like(anchor)
        support[3:6, 3:6, 3:6] = True
        support[9:11, 9:11, 9:11] = True
        result = transform(anchor, support, (1.0, 1.0, 1.0), "touching_component")
        self.assertTrue(result[3:6, 3:6, 3:6].all())
        self.assertFalse(result[9:11, 9:11, 9:11].any())


    def test_identity_is_exact(self):
        rng = np.random.default_rng(41)
        anchor = rng.random((8, 9, 10)) > 0.8
        support = rng.random((8, 9, 10)) > 0.5
        self.assertTrue(
            np.array_equal(transform(anchor, support, (1, 1, 1), "identity"), anchor)
        )


if __name__ == "__main__":
    unittest.main()
