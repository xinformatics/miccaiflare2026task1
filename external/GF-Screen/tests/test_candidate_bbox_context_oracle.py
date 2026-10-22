import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from analyze_candidate_bbox_context_oracle import analyze_case, cohort_summary


class CandidateBoundingBoxContextOracleTests(unittest.TestCase):
    def test_multiple_proposals_are_unioned(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target_path = root / "target.nii.gz"
            first_path = root / "first.nii.gz"
            second_path = root / "second.nii.gz"
            target = np.zeros((7, 7, 7), dtype=np.uint8)
            target[1:3, 1, 1] = 1
            target[5, 5, 5] = 1
            first = np.zeros_like(target)
            first[1, 1, 1] = 1
            second = np.zeros_like(target)
            second[5, 5, 5] = 1
            for path, array in ((target_path, target), (first_path, first), (second_path, second)):
                nib.save(nib.Nifti1Image(array, np.eye(4)), path)
            row = analyze_case(
                (
                    "case.nii.gz",
                    str(target_path),
                    (str(first_path), str(second_path)),
                    0.0,
                    100.0,
                    72.0,
                    (0.0,),
                )
            )
            self.assertEqual(row["positive_candidates"], 2)
            self.assertAlmostEqual(row["margin_000_dsc"], 0.8)
            self.assertEqual(len(row["positive_component_mm3"]), 2)
            self.assertEqual(row["positive_component_seed_overlap_fraction"], [1.0, 1.0])

    def test_source_separate_policy_does_not_connect_touching_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target_path = root / "target.nii.gz"
            first_path = root / "first.nii.gz"
            second_path = root / "second.nii.gz"
            target = np.zeros((8, 8, 8), dtype=np.uint8)
            target[2:4, 3:5, 3:5] = 1
            target[4:6, 3:5, 3:5] = 1
            first = np.zeros_like(target)
            second = np.zeros_like(target)
            first[2:4, 3:5, 3:5] = 1
            second[4:6, 3:5, 3:5] = 1
            for path, array in (
                (target_path, target),
                (first_path, first),
                (second_path, second),
            ):
                nib.save(nib.Nifti1Image(array, np.eye(4)), path)

            union_row = analyze_case(
                (
                    "case.nii.gz",
                    str(target_path),
                    (str(first_path), str(second_path)),
                    0.0,
                    100.0,
                    72.0,
                    (0.0,),
                    "union",
                )
            )
            separate_row = analyze_case(
                (
                    "case.nii.gz",
                    str(target_path),
                    (str(first_path), str(second_path)),
                    0.0,
                    100.0,
                    72.0,
                    (0.0,),
                    "source_separate",
                )
            )

            self.assertEqual(union_row["positive_candidates"], 1)
            self.assertEqual(separate_row["positive_candidates"], 2)
            self.assertEqual(union_row["margin_000_tiles"], 1)
            self.assertEqual(separate_row["margin_000_tiles"], 2)
            self.assertEqual(union_row["margin_000_dsc"], 1.0)
            self.assertEqual(separate_row["margin_000_dsc"], 1.0)

    def test_cohort_summary_requires_exact_coverage(self):
        rows = [
            {
                "case": "case0",
                "positive_candidates": 2,
                "positive_component_mm3": [10.0, 20.0],
                "positive_component_seed_overlap_fraction": [0.5, 1.0],
                "margin_024_dsc": 0.8,
                "margin_024_tiles": 3,
            },
            {
                "case": "case1",
                "positive_candidates": 1,
                "positive_component_mm3": [30.0],
                "positive_component_seed_overlap_fraction": [0.75],
                "margin_024_dsc": 0.6,
                "margin_024_tiles": 1,
            },
        ]
        result = cohort_summary(rows, {"case0": "A", "case1": "A"}, (24.0,))
        self.assertEqual(result["A"]["cases"], 2)
        self.assertEqual(result["A"]["positive_candidates"], 3)
        self.assertAlmostEqual(result["A"]["margins"]["24"]["macro_dsc"], 0.7)
        with self.assertRaises(ValueError):
            cohort_summary(rows, {"case0": "A"}, (24.0,))

    def test_margin_recovers_touched_target_beyond_seed_box(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target_path = root / "target.nii.gz"
            proposal_path = root / "proposal.nii.gz"
            affine = np.eye(4)
            target = np.zeros((7, 7, 7), dtype=np.uint8)
            target[2:4, 2, 2] = 1
            proposal = np.zeros_like(target)
            proposal[2, 2, 2] = 1
            nib.save(nib.Nifti1Image(target, affine), target_path)
            nib.save(nib.Nifti1Image(proposal, affine), proposal_path)

            row = analyze_case(
                (
                    "case.nii.gz",
                    str(target_path),
                    str(proposal_path),
                    0.0,
                    100.0,
                    72.0,
                    (0.0, 1.0),
                )
            )

            self.assertEqual(row["positive_candidates"], 1)
            self.assertAlmostEqual(row["margin_000_dsc"], 2 / 3)
            self.assertEqual(row["margin_001_dsc"], 1.0)
            self.assertEqual(row["margin_000_tiles"], 1)
            self.assertEqual(row["margin_001_tiles"], 1)


if __name__ == "__main__":
    unittest.main()
