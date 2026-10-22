import csv
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from evaluate_public_grid import csv_is_complete, evaluate_grid


class EvaluatePublicGridTests(unittest.TestCase):
    def test_grid_writes_complete_exact_candidate_csvs_and_resumes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gt = root / "gt"
            left = root / "left"
            right = root / "right"
            output = root / "scores"
            for directory in (gt, left, right):
                directory.mkdir()
            affine = np.diag([1.0, 1.0, 2.0, 1.0])
            for index in range(2):
                name = f"case_{index}.nii.gz"
                target = np.zeros((9, 9, 9), dtype=np.uint8)
                target[2:5, 2:5, 2:5] = 1
                prediction_left = target.copy()
                prediction_right = np.zeros_like(target)
                nib.save(nib.Nifti1Image(target, affine), gt / name)
                nib.save(nib.Nifti1Image(prediction_left, affine), left / name)
                nib.save(nib.Nifti1Image(prediction_right, affine), right / name)

            paths = evaluate_grid(
                gt, {"left": left, "right": right}, output, workers=2
            )
            names = ["case_0.nii.gz", "case_1.nii.gz"]
            self.assertTrue(csv_is_complete(paths["left"], names))
            self.assertTrue(csv_is_complete(paths["right"], names))
            with paths["left"].open(newline="") as stream:
                left_rows = list(csv.DictReader(stream))
            with paths["right"].open(newline="") as stream:
                right_rows = list(csv.DictReader(stream))
            self.assertEqual([float(row["Lesion_DSC"]) for row in left_rows], [1.0, 1.0])
            self.assertEqual([float(row["Lesion_DSC"]) for row in right_rows], [0.0, 0.0])

            mtimes = {tag: path.stat().st_mtime_ns for tag, path in paths.items()}
            evaluate_grid(gt, {"left": left, "right": right}, output, workers=2)
            self.assertEqual(
                mtimes, {tag: path.stat().st_mtime_ns for tag, path in paths.items()}
            )


if __name__ == "__main__":
    unittest.main()
