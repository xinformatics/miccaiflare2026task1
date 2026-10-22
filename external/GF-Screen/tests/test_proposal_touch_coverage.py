import unittest

import numpy as np

from analyze_proposal_touch_coverage import aggregate, coverage_metrics, filter_union


class ProposalTouchCoverageTests(unittest.TestCase):
    def test_touch_reconstructs_whole_target_component(self):
        target = np.zeros((5, 5, 5), dtype=bool)
        target[1:3, 1, 1] = True
        target[4, 4, 4] = True
        components = np.zeros_like(target, dtype=np.uint8)
        components[1:3, 1, 1] = 1
        components[4, 4, 4] = 2
        proposal = np.zeros_like(target)
        proposal[1, 1, 1] = True
        result = coverage_metrics(proposal, target, components)
        self.assertAlmostEqual(result["proposal_voxel_recall"], 1 / 3)
        self.assertAlmostEqual(result["gt_component_reconstruction_oracle_dice"], 0.8)
        self.assertAlmostEqual(result["gt_component_recall"], 0.5)

    def test_volume_filter(self):
        union = np.zeros((4, 4, 4), dtype=bool)
        union[0, 0, 0] = True
        union[2:4, 2:4, 2:4] = True
        filtered = filter_union(union, voxel_mm3=1.0, minimum_mm3=5.0)
        self.assertEqual(int(filtered.sum()), 8)

    def test_aggregate_checks_manifest_and_cohorts(self):
        row = {
            "case": "case",
            "proposal_voxel_recall": 0.5,
            "voxel_fp_free_oracle_dice": 0.6,
            "gt_component_reconstruction_oracle_dice": 0.7,
            "gt_component_recall": 0.8,
            "source_metrics": {
                "one": {
                    "proposal_voxel_recall": 0.5,
                    "voxel_fp_free_oracle_dice": 0.6,
                    "gt_component_reconstruction_oracle_dice": 0.7,
                    "gt_component_recall": 0.8,
                }
            },
        }
        result = aggregate([row], (("one", "/tmp/one"),), {"case": "cohort"})
        self.assertEqual(result["cohort_summary"]["cohort"]["cases"], 1)
        with self.assertRaises(ValueError):
            aggregate([row], (("one", "/tmp/one"),), {"other": "cohort"})


if __name__ == "__main__":
    unittest.main()
