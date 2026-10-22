import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from audit_nested_teacher_masks import audit, parse_threshold_dirs


def save(path: Path, data: np.ndarray, affine: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(data.astype(np.uint8), np.eye(4) if affine is None else affine),
        str(path),
    )


class AuditNestedTeacherMasksTest(unittest.TestCase):
    def test_nested_geometry_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = np.zeros((4, 4, 4), dtype=np.uint8)
            save(root / "images" / "case_0000.nii.gz", source)
            low = source.copy()
            low[1:3, 1:3, 1:3] = 1
            high = source.copy()
            high[2, 2, 2] = 1
            save(root / "low" / "case.nii.gz", low)
            save(root / "high" / "case.nii.gz", high)

            payload = audit(
                root / "images", [(0.5, root / "low"), (0.9, root / "high")], 10
            )

            self.assertEqual(payload["status"], "PASS")
            self.assertEqual(payload["audited_cases"], 1)

    def test_non_nested_mask_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = np.zeros((4, 4, 4), dtype=np.uint8)
            save(root / "images" / "case_0000.nii.gz", source)
            low = source.copy()
            low[1, 1, 1] = 1
            high = source.copy()
            high[2, 2, 2] = 1
            save(root / "low" / "case.nii.gz", low)
            save(root / "high" / "case.nii.gz", high)

            payload = audit(
                root / "images", [(0.5, root / "low"), (0.9, root / "high")], 10
            )

            self.assertEqual(payload["status"], "FAIL")
            self.assertTrue(any("not nested" in error for error in payload["errors"]))

    def test_streamed_slabs_preserve_counts_and_detect_late_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = np.zeros((4, 5, 9), dtype=np.uint8)
            save(root / "images" / "case_0000.nii.gz", source)
            low = source.copy()
            low[1:3, 1:4, 1:8] = 1
            high = source.copy()
            high[2, 2, 7] = 1
            save(root / "low" / "case.nii.gz", low)
            save(root / "high" / "case.nii.gz", high)

            payload = audit(
                root / "images",
                [(0.5, root / "low"), (0.9, root / "high")],
                10,
                slab_depth=2,
            )

            self.assertEqual(payload["status"], "PASS")
            self.assertEqual(
                payload["threshold_stats"]["0.5"]["foreground_voxels"], 42
            )
            self.assertEqual(
                payload["threshold_stats"]["0.9"]["foreground_voxels"], 1
            )

            high[0, 0, 8] = 1
            save(root / "high" / "case.nii.gz", high)
            failed = audit(
                root / "images",
                [(0.5, root / "low"), (0.9, root / "high")],
                10,
                slab_depth=2,
            )
            self.assertEqual(failed["status"], "FAIL")
            self.assertTrue(any("not nested" in error for error in failed["errors"]))

    def test_slab_depth_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            audit(Path("/unused"), [(0.5, Path("/low")), (0.9, Path("/high"))], 1, 0)

    def test_threshold_parser_sorts_and_rejects_duplicates(self) -> None:
        parsed = parse_threshold_dirs(["0.9=/high", "0.5=/low"])
        self.assertEqual([value[0] for value in parsed], [0.5, 0.9])
        with self.assertRaises(ValueError):
            parse_threshold_dirs(["0.5=/a", "0.5=/b"])


if __name__ == "__main__":
    unittest.main()
