import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from audit_selected_weak_prompt_bank import audit


class SelectedWeakPromptBankAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "inputs"
        self.images = self.root / "images"
        self.predictions = self.root / "predictions"
        self.nifti = self.root / "nifti"
        for directory in (self.inputs, self.images, self.predictions, self.nifti):
            directory.mkdir()
        self.plan = self.root / "plan.json"
        self.plan.write_text(
            json.dumps(
                {
                    "prompts": {"autopet": ["tumor"], "lndb": ["lung nodule"]},
                    "existence_threshold": 0.2,
                    "disable_nms": True,
                }
            )
        )
        affine = np.diag([1.5, 1.5, 1.5, 1.0])
        for case in ("a", "b"):
            np.savez_compressed(self.inputs / f"{case}_0000.npz", image=np.zeros((2, 2)))
            np.savez_compressed(
                self.predictions / f"{case}_0000.npz",
                prompts=np.asarray(["tumor"]),
                segs_e020=np.zeros((2, 2, 2), dtype=np.uint8),
            )
            nib.save(
                nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.float32), affine),
                self.images / f"{case}_0000.nii.gz",
            )
            nib.save(
                nib.Nifti1Image(np.zeros((2, 2, 2), dtype=np.uint8), affine),
                self.nifti / f"{case}.nii.gz",
            )

    def tearDown(self):
        self.temporary.cleanup()

    def test_exact_bank_passes(self):
        report = audit(
            self.inputs, self.images, self.predictions, self.nifti,
            self.plan, "autopet", 2,
        )
        self.assertEqual(report["cases"], 2)
        self.assertTrue(report["all_masks_finite_binary"])
        self.assertFalse(report["labels_read"])

    def test_prompt_mismatch_fails(self):
        path = self.predictions / "a_0000.npz"
        np.savez_compressed(
            path,
            prompts=np.asarray(["wrong"]),
            segs_e020=np.zeros((2, 2, 2), dtype=np.uint8),
        )
        with self.assertRaisesRegex(ValueError, "prompt mismatch"):
            audit(
                self.inputs, self.images, self.predictions, self.nifti,
                self.plan, "autopet", 2,
            )


if __name__ == "__main__":
    unittest.main()
