import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from audit_proposal_union import audit_union


class ProposalUnionAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for name in ("images", "a", "b", "union"):
            (self.root / name).mkdir()
        self.affine = np.diag([1.0, 1.5, 2.0, 1.0])
        self.case = "case001"
        self.prediction = f"{self.case}.nii.gz"
        image = np.zeros((5, 6, 7), dtype=np.int16)
        nib.save(
            nib.Nifti1Image(image, self.affine),
            self.root / "images" / f"{self.case}_0000.nii.gz",
        )
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "items": [
                        {
                            "name": self.case,
                            "prediction_name": self.prediction,
                            "image": str(
                                self.root / "images" / f"{self.case}_0000.nii.gz"
                            ),
                            "complete_label": True,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def save_mask(self, directory: str, mask: np.ndarray) -> None:
        nib.save(
            nib.Nifti1Image(mask.astype(np.uint8), self.affine),
            self.root / directory / self.prediction,
        )

    def test_exact_union_passes_with_provenance(self):
        a = np.zeros((5, 6, 7), dtype=bool)
        b = np.zeros_like(a)
        a[1, 2, 3] = True
        b[2, 3, 4] = True
        self.save_mask("a", a)
        self.save_mask("b", b)
        self.save_mask("union", a | b)
        report = audit_union(
            self.manifest,
            self.root / "a",
            self.root / "b",
            self.root / "union",
            workers=1,
        )
        self.assertTrue(report["all_unions_exact_voxelwise"])
        self.assertFalse(report["labels_read"])
        self.assertEqual(report["voxel_totals"]["union"], 2)
        self.assertEqual(len(report["inventories"]["union"]["sha256_inventory"]), 64)

    def test_voxelwise_mismatch_is_rejected(self):
        a = np.zeros((5, 6, 7), dtype=bool)
        b = np.zeros_like(a)
        a[1, 2, 3] = True
        b[2, 3, 4] = True
        self.save_mask("a", a)
        self.save_mask("b", b)
        self.save_mask("union", a)
        with self.assertRaisesRegex(ValueError, "union mismatch"):
            audit_union(
                self.manifest,
                self.root / "a",
                self.root / "b",
                self.root / "union",
                workers=1,
            )

    def test_extra_file_is_rejected(self):
        empty = np.zeros((5, 6, 7), dtype=bool)
        self.save_mask("a", empty)
        self.save_mask("b", empty)
        self.save_mask("union", empty)
        nib.save(
            nib.Nifti1Image(empty.astype(np.uint8), self.affine),
            self.root / "union" / "extra.nii.gz",
        )
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            audit_union(
                self.manifest,
                self.root / "a",
                self.root / "b",
                self.root / "union",
                workers=1,
            )

    def test_unlabeled_mode_is_explicit(self):
        payload = json.loads(self.manifest.read_text(encoding="utf-8"))
        payload["items"][0]["complete_label"] = False
        self.manifest.write_text(json.dumps(payload), encoding="utf-8")
        empty = np.zeros((5, 6, 7), dtype=bool)
        self.save_mask("a", empty)
        self.save_mask("b", empty)
        self.save_mask("union", empty)
        with self.assertRaisesRegex(ValueError, "complete labels"):
            audit_union(
                self.manifest, self.root / "a", self.root / "b",
                self.root / "union", workers=1,
            )
        report = audit_union(
            self.manifest, self.root / "a", self.root / "b",
            self.root / "union", workers=1, require_complete_labels=False,
        )
        self.assertFalse(report["complete_labels_required"])
        self.assertFalse(report["labels_read"])


if __name__ == "__main__":
    unittest.main()
