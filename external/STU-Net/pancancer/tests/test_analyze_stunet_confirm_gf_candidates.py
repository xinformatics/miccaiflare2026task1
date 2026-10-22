import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "analyze_stunet_confirm_gf_candidates.py"
SPEC = importlib.util.spec_from_file_location("confirm_screen", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ConfirmationCountsTest(unittest.TestCase):
    def test_keeps_only_disconnected_large_supported_components(self):
        shape = (24, 24, 24)
        candidate = np.zeros(shape, dtype=bool)
        anchor = np.zeros(shape, dtype=bool)
        support = np.zeros(shape, dtype=bool)
        target = np.zeros(shape, dtype=bool)

        # 4^3 voxels at 50 mm3/voxel = 3,200 mm3, true and 12.5% supported.
        candidate[1:5, 1:5, 1:5] = True
        support[1:3, 1:3, 1:3] = True
        target[1:5, 1:5, 1:5] = True

        # Another 3,200-mm3 component, false and 1.5625% supported.
        candidate[10:14, 10:14, 10:14] = True
        support[10, 10, 10] = True

        # Large and supported, but touching the immutable anchor: ineligible.
        candidate[18:22, 18:22, 18:22] = True
        support[18:22, 18:22, 18:22] = True
        anchor[18, 18, 18] = True
        target[18, 18, 18] = True

        result = MODULE.component_confirmation_counts(
            candidate=candidate,
            anchor=anchor,
            support=support,
            target=target,
            voxel_mm3=50.0,
        )
        self.assertEqual(result.selected_components["stu_s01"], 2)
        self.assertEqual(result.true_selected_components["stu_s01"], 1)
        self.assertEqual(result.selected_components["stu_s10"], 1)
        self.assertEqual(result.true_selected_components["stu_s10"], 1)
        self.assertGreater(result.candidate_dsc["stu_s10"], result.anchor_dsc)
        self.assertLess(result.candidate_dsc["stu_s01"], result.candidate_dsc["stu_s10"])

    def test_six_connectivity_does_not_merge_corner_touch(self):
        candidate = np.zeros((12, 12, 12), dtype=bool)
        anchor = np.zeros_like(candidate)
        support = np.zeros_like(candidate)
        target = np.zeros_like(candidate)
        candidate[1:5, 1:5, 1:5] = True
        candidate[5:9, 5:9, 5:9] = True
        support[1:5, 1:5, 1:5] = True
        target[1:5, 1:5, 1:5] = True
        result = MODULE.component_confirmation_counts(
            candidate=candidate,
            anchor=anchor,
            support=support,
            target=target,
            voxel_mm3=50.0,
        )
        self.assertEqual(result.selected_components["stu_s01"], 1)

    def test_empty_candidate_is_anchor_identity(self):
        shape = (8, 8, 8)
        candidate = np.zeros(shape, dtype=bool)
        anchor = np.zeros(shape, dtype=bool)
        support = np.zeros(shape, dtype=bool)
        target = np.zeros(shape, dtype=bool)
        anchor[2:4, 2:4, 2:4] = True
        target[2:4, 2:4, 2:4] = True
        result = MODULE.component_confirmation_counts(
            candidate=candidate,
            anchor=anchor,
            support=support,
            target=target,
            voxel_mm3=1.0,
        )
        self.assertEqual(result.anchor_dsc, 1.0)
        self.assertEqual(result.candidate_dsc, {"stu_s01": 1.0, "stu_s10": 1.0})


if __name__ == "__main__":
    unittest.main()
