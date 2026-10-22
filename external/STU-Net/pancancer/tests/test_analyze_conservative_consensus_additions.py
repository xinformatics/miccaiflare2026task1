import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "analyze_conservative_consensus_additions.py"
SPEC = importlib.util.spec_from_file_location("consensus_additions", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ConservativeConsensusAdditionTest(unittest.TestCase):
    def test_only_adds_large_absent_component_with_both_supports(self):
        shape = (30, 30, 30)
        proposal = np.zeros(shape, dtype=bool)
        anchor = np.zeros(shape, dtype=bool)
        peer = np.zeros(shape, dtype=bool)
        independent = np.zeros(shape, dtype=bool)
        target = np.zeros(shape, dtype=bool)
        anchor[25:27, 25:27, 25:27] = True
        target[25:27, 25:27, 25:27] = True

        # 4^3 * 50 mm3 = 3,200 mm3; 50% peer and 12.5% independent support.
        proposal[1:5, 1:5, 1:5] = True
        peer[1:3, 1:5, 1:5] = True
        independent[1:3, 1:3, 1:3] = True
        target[1:5, 1:5, 1:5] = True

        # Large but only 25% peer support: rejected.
        proposal[10:14, 10:14, 10:14] = True
        peer[10:11, 10:14, 10:14] = True
        independent[10:14, 10:14, 10:14] = True

        result = MODULE.case_counts_from_arrays(
            proposal=proposal,
            anchor=anchor,
            peer=peer,
            independent=independent,
            target=target,
            voxel_mm3=50.0,
        )
        self.assertEqual(result.selected_components, 1)
        self.assertEqual(result.true_selected_components, 1)
        self.assertGreater(result.candidate_dsc, result.anchor_dsc)

    def test_empty_proposal_is_anchor_identity(self):
        shape = (8, 8, 8)
        proposal = np.zeros(shape, dtype=bool)
        anchor = np.zeros(shape, dtype=bool)
        target = np.zeros(shape, dtype=bool)
        anchor[2:4, 2:4, 2:4] = True
        target[2:4, 2:4, 2:4] = True
        result = MODULE.case_counts_from_arrays(
            proposal=proposal,
            anchor=anchor,
            peer=np.zeros(shape, dtype=bool),
            independent=np.zeros(shape, dtype=bool),
            target=target,
            voxel_mm3=1.0,
        )
        self.assertEqual(result.anchor_dsc, 1.0)
        self.assertEqual(result.candidate_dsc, 1.0)
        self.assertEqual(result.selected_components, 0)


if __name__ == "__main__":
    unittest.main()
