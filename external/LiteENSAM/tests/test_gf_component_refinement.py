import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from refine_gf_components import dice_pair, longest_diameter_prompt, window_to_uint8


class TestGFComponentRefinement(unittest.TestCase):
    def test_diameter_is_on_largest_slice_and_inside_component(self):
        component = np.zeros((5, 12, 14), dtype=bool)
        component[1, 3:5, 4:7] = True
        component[3, 2:8, 3:11] = True
        prompt = longest_diameter_prompt(component)
        self.assertEqual(prompt.shape, (2, 3))
        self.assertTrue(np.all(prompt[:, 0] == 3))
        for z, y, x in prompt.astype(int):
            self.assertTrue(component[z, y, x])

    def test_single_voxel_prompt_is_valid(self):
        component = np.zeros((3, 4, 5), dtype=bool)
        component[1, 2, 3] = True
        prompt = longest_diameter_prompt(component)
        np.testing.assert_array_equal(prompt[0], [1, 2, 3])
        np.testing.assert_array_equal(prompt[1], [1, 2, 3])

    def test_window_is_clipped_uint8(self):
        image = np.asarray([-200.0, -175.0, 37.5, 250.0, 400.0])
        result = window_to_uint8(image, -175, 250)
        self.assertEqual(result.dtype, np.uint8)
        np.testing.assert_array_equal(result, [0, 0, 128, 255, 255])

    def test_pair_dice(self):
        first = np.asarray([1, 1, 0, 0], dtype=bool)
        second = np.asarray([0, 1, 1, 0], dtype=bool)
        self.assertAlmostEqual(dice_pair(first, second), 0.5)
        self.assertEqual(dice_pair(np.zeros(1, bool), np.zeros(1, bool)), 1.0)


if __name__ == "__main__":
    unittest.main()
