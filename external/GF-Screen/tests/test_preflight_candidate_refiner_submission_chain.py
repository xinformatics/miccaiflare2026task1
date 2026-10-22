import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from preflight_candidate_refiner_submission_chain import (
    atomic_json,
    audit_cells,
    inventory,
    same_cases,
    zip_inventory,
)


class CandidateRefinerSubmissionPreflightTests(unittest.TestCase):
    def test_inventory_normalizes_modality_suffix_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for case in ("case001", "case002"):
                (root / f"{case}_0000.nii.gz").write_bytes(b"not-empty")
            images = inventory(root, "_0000.nii.gz", "_0000.nii.gz", 2)
            self.assertEqual(images["cases"], ["case001", "case002"])
            with self.assertRaisesRegex(SystemExit, "expected 3"):
                inventory(root, "_0000.nii.gz", "_0000.nii.gz", 3)

    def test_case_alignment_fails_closed(self):
        reference = {"cases": ["case001", "case002"]}
        candidate = {"cases": ["case001", "case003"]}
        with self.assertRaisesRegex(SystemExit, "inventory mismatch"):
            same_cases(reference, {"candidate": candidate}, "hidden")

    def test_zip_requires_flat_unique_nonempty_masks_and_crc(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = root / "valid.zip"
            with zipfile.ZipFile(valid, "w") as archive:
                archive.writestr("case001.nii.gz", b"mask-one")
                archive.writestr("case002.nii.gz", b"mask-two")
            result = zip_inventory(valid, 2)
            self.assertEqual(result["crc"], "PASS")
            self.assertTrue(result["flat"])

            nested = root / "nested.zip"
            with zipfile.ZipFile(nested, "w") as archive:
                archive.writestr("predictions/case001.nii.gz", b"mask")
            with self.assertRaisesRegex(SystemExit, "not flat"):
                zip_inventory(nested, 1)

    def test_cell_marker_binds_geometry_and_label_blindness(self):
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "PROPOSALS_COMPLETE.json"
            payload = {
                "method": "filled_48mm_occupied_lung_cells_in_eight_parity_sources",
                "cases": 2,
                "sources": 8,
                "parameters": {"target_spacing_mm": 2.5, "cell_stride_mm": 48.0},
                "maximum_component_axis_extent_mm": 48.0,
                "labels_read": False,
                "complete_labels_required": False,
                "rows": [{"case": "case001.nii.gz"}, {"case": "case002.nii.gz"}],
            }
            marker.write_text(json.dumps(payload))
            result = audit_cells(marker, 2)
            self.assertTrue(all(result["checks"].values()))
            payload["labels_read"] = True
            marker.write_text(json.dumps(payload))
            with self.assertRaisesRegex(SystemExit, "invalid lung-cell marker"):
                audit_cells(marker, 2)

    def test_atomic_json_replaces_complete_document(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "audit.json"
            atomic_json(output, {"status": "PASS", "count": 100})
            self.assertEqual(json.loads(output.read_text()), {"status": "PASS", "count": 100})
            self.assertEqual(list(output.parent.glob(f".{output.name}.*")), [])


if __name__ == "__main__":
    unittest.main()
