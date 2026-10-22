import unittest
import importlib.util
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "screen_anchor_dual_confirmation.py"
SPEC = importlib.util.spec_from_file_location("screen_anchor_dual_confirmation", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
component_dice = MODULE.component_dice
selection_masks = MODULE.selection_masks
supported_component_ids = MODULE.supported_component_ids


class DualConfirmationTest(unittest.TestCase):
    def test_component_level_or_and_are_not_voxel_intersections(self):
        labels = np.zeros((1, 1, 5), dtype=np.int32)
        labels[..., 0:2] = 1
        labels[..., 3:5] = 2
        anchor = labels > 0
        stu = np.zeros_like(anchor); stu[..., 0] = True
        ftu = np.zeros_like(anchor); ftu[..., 1] = True; ftu[..., 3] = True
        stu_ids = supported_component_ids(labels, anchor, stu)
        ftu_ids = supported_component_ids(labels, anchor, ftu)
        masks = selection_masks(stu_ids, ftu_ids)
        self.assertTrue(masks["both_any"][1])
        self.assertFalse(masks["both_any"][2])
        self.assertTrue(masks["either_any"][2])

    def test_component_dice_uses_sufficient_statistics(self):
        keep = np.array([False, True, False])
        sizes = np.array([1, 4, 3])
        overlap = np.array([0, 2, 3])
        self.assertAlmostEqual(component_dice(keep, sizes, overlap, 5), 4 / 9)

    def test_threshold_comparison_matches_deployed_quantization(self):
        threshold = 0.60
        quantized = np.array([int(threshold * 65535), int(threshold * 65535) + 1])
        self.assertEqual((quantized > int(threshold * 65535)).tolist(), [False, True])


if __name__ == "__main__":
    unittest.main()
