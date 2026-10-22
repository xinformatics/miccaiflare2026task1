import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from audit_training_proposal_sources import audit_sources, parse_sources


class AuditTrainingProposalSourcesTests(unittest.TestCase):
    def _fixture(self, root: Path):
        image_dir = root / "images"
        a_dir = root / "a"
        b_dir = root / "b"
        for directory in (image_dir, a_dir, b_dir):
            directory.mkdir()
        items = []
        affine = np.diag([1.0, 1.0, 2.0, 1.0])
        for index in range(2):
            name = f"case{index}.nii.gz"
            image = image_dir / name
            nib.save(nib.Nifti1Image(np.zeros((3, 4, 5), dtype=np.float32), affine), image)
            for directory in (a_dir, b_dir):
                mask = np.zeros((3, 4, 5), dtype=np.uint8)
                mask[index, 1, 1] = 1
                nib.save(nib.Nifti1Image(mask, affine), directory / name)
            items.append({"prediction_name": name, "image": str(image), "complete_label": True})
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"items": items}), encoding="utf-8")
        return manifest, a_dir, b_dir

    def test_exact_binary_geometry_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, a_dir, b_dir = self._fixture(Path(temporary))
            report = audit_sources(
                manifest,
                [("a", a_dir.resolve()), ("b", b_dir.resolve())],
                workers=1,
                expected_count=2,
            )
            self.assertEqual(report["cases"], 2)
            self.assertEqual(report["source_names"], ["a", "b"])
            self.assertTrue(report["all_masks_finite_binary"])
            self.assertEqual(report["inventories"]["a"]["nonempty_cases"], 2)

    def test_geometry_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, a_dir, b_dir = self._fixture(Path(temporary))
            path = b_dir / "case1.nii.gz"
            nib.save(
                nib.Nifti1Image(np.zeros((2, 4, 5), dtype=np.uint8), np.eye(4)),
                path,
            )
            with self.assertRaisesRegex(ValueError, "geometry mismatch"):
                audit_sources(manifest, [("a", a_dir), ("b", b_dir)], 1, 2)

    def test_duplicate_source_name_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, a_dir, b_dir = self._fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                parse_sources([f"same={a_dir}", f"same={b_dir}"])


if __name__ == "__main__":
    unittest.main()
