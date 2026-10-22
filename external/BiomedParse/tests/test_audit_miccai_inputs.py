import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from audit_miccai_inputs import audit_inputs
from miccai_nifti_adapter import prepare_case


class MiccaiInputAuditTests(unittest.TestCase):
    def test_prepared_input_passes_and_prompt_change_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images, inputs = root / "images", root / "inputs"
            images.mkdir(); inputs.mkdir()
            image_path = images / "case001_0000.nii.gz"
            image = sitk.GetImageFromArray(
                np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6)
            )
            image.SetSpacing((0.8, 0.9, 2.0))
            sitk.WriteImage(image, str(image_path), True)
            prompts = ("lung tumor", "lung nodule")
            prepare_case(image_path, inputs / "case001_0000.npz", list(prompts), -160, 240)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"items": [{
                    "name": "case001", "image": str(image_path),
                    "cohort": "chest:Chest_LNDb", "complete_label": True,
                }]}), encoding="utf-8",
            )
            report = audit_inputs(
                manifest, "chest:Chest_LNDb", images, inputs, prompts, 1, 1
            )
            self.assertTrue(report["all_prompt_maps_match"])
            self.assertFalse(report["labels_read"])
            with self.assertRaisesRegex(ValueError, "prompt mismatch"):
                audit_inputs(
                    manifest, "chest:Chest_LNDb", images, inputs,
                    ("lung tumor",), 1, 1,
                )


if __name__ == "__main__":
    unittest.main()
