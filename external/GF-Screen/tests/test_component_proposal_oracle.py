import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from analyze_component_proposal_oracle import analyze_case, summarize_cohorts


class ComponentProposalOracleTests(unittest.TestCase):
    def test_cohort_summary(self):
        rows = [
            {
                "case": "a",
                "proposal_voxel_recall": 0.5,
                "voxel_fp_free_oracle_dice": 0.6,
                "gt_component_reconstruction_oracle_dice": 0.7,
            },
            {
                "case": "b",
                "proposal_voxel_recall": 0.7,
                "voxel_fp_free_oracle_dice": 0.8,
                "gt_component_reconstruction_oracle_dice": 0.9,
            },
        ]
        result = summarize_cohorts(rows, {"a": "one", "b": "one"})
        self.assertAlmostEqual(result["one"]["proposal_voxel_recall"], 0.6)
        self.assertEqual(result["one"]["cases"], 2)
        with self.assertRaises(ValueError):
            summarize_cohorts(rows, {"a": "one"})

    def test_component_precision_bits_and_voxel_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gt_dir = root / "gt"
            source_a = root / "a"
            source_b = root / "b"
            for directory in (gt_dir, source_a, source_b):
                directory.mkdir()
            name = "case.nii.gz"
            affine = np.eye(4)
            target = np.zeros((5, 5, 5), dtype=np.uint8)
            target[1, 1, 1] = 1
            target[3, 3, 3] = 1
            prediction_a = np.zeros_like(target)
            prediction_a[1, 1, 1] = 1
            prediction_a[1, 1, 2] = 1  # one connected false-positive voxel
            prediction_b = np.zeros_like(target)
            prediction_b[3, 3, 3] = 1
            for directory, array in (
                (gt_dir, target),
                (source_a, prediction_a),
                (source_b, prediction_b),
            ):
                nib.save(nib.Nifti1Image(array, affine), directory / name)

            row = analyze_case(
                (
                    name,
                    str(gt_dir / name),
                    (("a", str(source_a)), ("b", str(source_b))),
                )
            )
            self.assertEqual(row["proposal_voxel_recall"], 1.0)
            self.assertEqual(row["voxel_fp_free_oracle_dice"], 1.0)
            self.assertEqual(row["gt_component_reconstruction_oracle_dice"], 1.0)
            self.assertEqual(row["detected_gt_components"], 2)
            self.assertAlmostEqual(row["component_precision_0.5_dice"], 0.8)
            self.assertEqual(row["component_precision_0.5_voxels"], 3)
            self.assertAlmostEqual(
                row["source_metrics"]["a"]["proposal_voxel_recall"], 0.5
            )
            self.assertAlmostEqual(
                row["source_metrics"]["a"]["voxel_fp_free_oracle_dice"], 2 / 3
            )
            self.assertAlmostEqual(
                row["source_metrics"]["a"][
                    "leave_one_out_voxel_fp_free_oracle_dice"
                ],
                2 / 3,
            )
            self.assertAlmostEqual(
                row["source_metrics"]["b"][
                    "leave_one_out_gt_component_reconstruction_oracle_dice"
                ],
                2 / 3,
            )


if __name__ == "__main__":
    unittest.main()
