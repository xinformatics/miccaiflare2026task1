import unittest

import numpy as np

from screen_touching_component_risk import transform


class TouchingComponentRiskTests(unittest.TestCase):
    def test_touch_all_reproduces_whole_touching_component(self):
        anchor = np.zeros((7, 7, 7), dtype=bool)
        anchor[2, 2, 2] = True
        support = np.zeros_like(anchor)
        support[2:5, 2, 2] = True
        support[6, 6, 6] = True
        result, audit = transform(anchor, support, 1.0, "touch_all")
        self.assertEqual(int(result.sum()), 3)
        self.assertEqual(audit["touching_components"], 1)
        self.assertEqual(audit["accepted_components"], 1)
        self.assertEqual(audit["added_voxels"], 2)

    def test_overlap_and_extension_gates_reject_risky_contact(self):
        anchor = np.zeros((8, 8, 8), dtype=bool)
        anchor[1, 1, 1] = True
        support = np.zeros_like(anchor)
        support[1:8, 1, 1] = True
        by_size, _ = transform(anchor, support, 1000.0, "added_mm3_5000")
        by_ratio, _ = transform(anchor, support, 1.0, "extension_ratio_0p50")
        self.assertTrue(np.array_equal(by_size, anchor))
        self.assertTrue(np.array_equal(by_ratio, anchor))

    def test_anchor_is_immutable_and_import_is_support_bounded(self):
        anchor = np.zeros((5, 5, 5), dtype=bool)
        anchor[1:3, 1:3, 1:3] = True
        support = np.zeros_like(anchor)
        support[2:4, 2:4, 2:4] = True
        result, _ = transform(anchor, support, 2.0, "added_mm3_5000")
        self.assertTrue(np.all(result[anchor]))
        self.assertFalse(np.any((result & ~anchor) & ~support))

    def test_invalid_inputs_fail_closed(self):
        mask = np.zeros((2, 2, 2), dtype=bool)
        with self.assertRaises(ValueError):
            transform(mask, np.zeros((3, 3, 3), dtype=bool), 1.0, "touch_all")
        with self.assertRaises(ValueError):
            transform(mask, mask, 0.0, "touch_all")


if __name__ == "__main__":
    unittest.main()
