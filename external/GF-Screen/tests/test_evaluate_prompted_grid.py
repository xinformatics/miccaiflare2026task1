import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from evaluate_prompted_grid import evaluate_prompted_grid


class EvaluatePromptedGridTests(unittest.TestCase):
    def test_exact_confirmation_is_scored_without_materialized_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gt = root / "gt"
            support = root / "support"
            candidate = root / "candidate"
            for directory in (gt, support, candidate):
                directory.mkdir()
            affine = np.eye(4)
            target = np.zeros((8, 8, 8), dtype=np.uint8)
            target[1:3, 1:3, 1:3] = 1
            base = target.copy()
            base[5:7, 5:7, 5:7] = 1
            prompt = target.copy()
            name = "case.nii.gz"
            for directory, array in ((gt, target), (support, prompt), (candidate, base)):
                nib.save(nib.Nifti1Image(array, affine), directory / name)
            outputs = evaluate_prompted_grid(
                gt,
                support,
                {"kind": "confirm", "distance_mm": 0.0, "rescue_min_mm3": "inf"},
                {"candidate": candidate},
                root / "scores",
                workers=1,
            )
            values = outputs["candidate"].read_text()
            self.assertIn("case,1.0,1.0,8", values)
            self.assertFalse((root / "predictions").exists())


if __name__ == "__main__":
    unittest.main()
