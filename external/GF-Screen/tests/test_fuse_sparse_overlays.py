import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from fuse_sparse_overlays import fuse_sparse


class SparseOverlayTests(unittest.TestCase):
    def save(self, path: Path, mask: np.ndarray, affine=None):
        nib.save(
            nib.Nifti1Image(mask.astype(np.uint8), np.eye(4) if affine is None else affine),
            path,
        )

    def test_sparse_overlay_preserves_unmentioned_case_and_unions_subset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("base", "overlay", "output"):
                (root / name).mkdir()
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"items": [
                {"prediction_name": "a.nii.gz"},
                {"prediction_name": "b.nii.gz"},
            ]}), encoding="utf-8")
            base_a = np.zeros((4, 4, 4), bool); base_a[0, 0, 0] = True
            base_b = np.zeros_like(base_a); base_b[1, 1, 1] = True
            extra = np.zeros_like(base_a); extra[2, 2, 2] = True
            self.save(root / "base" / "a.nii.gz", base_a)
            self.save(root / "base" / "b.nii.gz", base_b)
            self.save(root / "overlay" / "a.nii.gz", extra)
            report = fuse_sparse(
                manifest, root / "base", (root / "overlay",), root / "output", 1
            )
            a = np.asanyarray(nib.load(root / "output" / "a.nii.gz").dataobj) > 0
            b = np.asanyarray(nib.load(root / "output" / "b.nii.gz").dataobj) > 0
            self.assertTrue(np.array_equal(a, base_a | extra))
            self.assertTrue(np.array_equal(b, base_b))
            self.assertEqual(report["cases_with_overlay"], 1)
            self.assertFalse(report["labels_read"])

    def test_unknown_overlay_case_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("base", "overlay", "output"):
                (root / name).mkdir()
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps({"items": [{"prediction_name": "a.nii.gz"}]}),
                encoding="utf-8",
            )
            empty = np.zeros((2, 2, 2), bool)
            self.save(root / "base" / "a.nii.gz", empty)
            self.save(root / "overlay" / "unknown.nii.gz", empty)
            with self.assertRaisesRegex(ValueError, "unknown cases"):
                fuse_sparse(
                    manifest, root / "base", (root / "overlay",), root / "output", 1
                )

    def test_empty_base_completes_sparse_source_without_unioning_base(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("base", "overlay", "output"):
                (root / name).mkdir()
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "items": [
                            {"prediction_name": "a.nii.gz"},
                            {"prediction_name": "b.nii.gz"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            base_a = np.zeros((4, 4, 4), bool)
            base_a[0, 0, 0] = True
            base_b = np.zeros_like(base_a)
            base_b[1, 1, 1] = True
            overlay = np.zeros_like(base_a)
            overlay[2, 2, 2] = True
            self.save(root / "base" / "a.nii.gz", base_a)
            self.save(root / "base" / "b.nii.gz", base_b)
            self.save(root / "overlay" / "a.nii.gz", overlay)

            report = fuse_sparse(
                manifest,
                root / "base",
                (root / "overlay",),
                root / "output",
                1,
                empty_base=True,
            )
            a = np.asanyarray(nib.load(root / "output" / "a.nii.gz").dataobj) > 0
            b = np.asanyarray(nib.load(root / "output" / "b.nii.gz").dataobj) > 0
            self.assertTrue(np.array_equal(a, overlay))
            self.assertFalse(b.any())
            self.assertTrue(report["empty_base"])
            self.assertEqual(report["operation"], "empty_or_sparse_overlays")


if __name__ == "__main__":
    unittest.main()
