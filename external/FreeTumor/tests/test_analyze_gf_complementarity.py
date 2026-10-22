import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from external.FreeTumor.analyze_gf_complementarity import (
    evaluate,
    filter_components,
    large_miss_recovery,
    oracle_supported_components,
)


class AnalyzeGFComplementarityTest(unittest.TestCase):
    def test_component_filter_uses_physical_volume(self):
        mask = np.zeros((8, 8, 8), dtype=bool)
        mask[0, 0, 0] = True
        mask[3:5, 3:5, 3:5] = True
        filtered = filter_components(mask, voxel_mm3=100.0, minimum_mm3=500.0)
        self.assertFalse(filtered[0, 0, 0])
        self.assertTrue(filtered[3:5, 3:5, 3:5].all())

    def test_component_filter_matches_six_connected_deployment(self):
        mask = np.zeros((4, 4, 4), dtype=bool)
        mask[0, 0, 0] = True
        mask[1, 1, 1] = True
        filtered = filter_components(mask, voxel_mm3=100.0, minimum_mm3=150.0)
        self.assertFalse(filtered.any())

    def test_oracle_support_keeps_whole_touched_candidate_component(self):
        candidate = np.zeros((8, 8, 8), dtype=bool)
        candidate[1:3, 1:3, 1:3] = True
        candidate[5:7, 5:7, 5:7] = True
        gt = np.zeros_like(candidate)
        gt[1, 1, 1] = True
        supported = oracle_supported_components(candidate, gt)
        self.assertTrue(supported[1:3, 1:3, 1:3].all())
        self.assertFalse(supported[5:7, 5:7, 5:7].any())

    def test_large_miss_requires_ten_percent_overlap_and_3000_mm3(self):
        gt = np.zeros((10, 10, 10), dtype=bool)
        gt[1:3, 1:3, 1:3] = True
        anchor = np.zeros_like(gt)
        candidate = np.zeros_like(gt)
        candidate[1, 1, 1] = True
        rows = large_miss_recovery(gt, anchor, candidate, voxel_mm3=500.0)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["volume_mm3"], 4000.0)
        self.assertTrue(rows[0]["candidate_recovered"])

    @staticmethod
    def save(path: Path, array: np.ndarray) -> None:
        affine = np.diag([10.0, 10.0, 10.0, 1.0])
        nib.save(nib.Nifti1Image(array.astype(np.uint8), affine), path)

    def test_public_evaluation_detects_closed_case_complementarity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            labels = root / "labels"
            anchor = root / "anchor"
            candidate = root / "candidate"
            for directory in (labels, anchor, candidate):
                directory.mkdir()

            for name, anchor_has_lesion, candidate_has_lesion in (
                ("FLARE23Ts_0009.nii.gz", False, True),
                ("FLARE23Ts_0011.nii.gz", True, False),
            ):
                gt = np.zeros((8, 8, 8), dtype=np.uint8)
                gt[2:4, 2:4, 2:4] = 1
                anchor_mask = gt if anchor_has_lesion else np.zeros_like(gt)
                candidate_mask = gt if candidate_has_lesion else np.zeros_like(gt)
                self.save(labels / name, gt)
                self.save(anchor / name, anchor_mask)
                self.save(candidate / name, candidate_mask)

            result = evaluate(labels, anchor, candidate)

        self.assertEqual(result["case_count"], 2)
        self.assertEqual(result["large_anchor_miss_count"], 1)
        self.assertEqual(result["large_misses_recovered_by_candidate"], 1)
        self.assertEqual(result["closed_gf_cases_recovered_by_candidate"], 1)
        self.assertEqual(result["recovered_closed_cases"], ["FLARE23Ts_0009"])
        self.assertTrue(result["independent_proposal_signal"])
        self.assertEqual(result["metrics"]["union_mm3000"]["Lesion_DSC"], 1.0)
        self.assertEqual(result["metrics"]["union_mm3000"]["Lesion_NSD"], 1.0)
        self.assertTrue(result["direct_union_public_gate"])


if __name__ == "__main__":
    unittest.main()
