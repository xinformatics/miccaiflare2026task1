import hashlib
import tempfile
import unittest
from pathlib import Path

from focus_composite_deployment_selection import select


class FocusCompositeDeploymentSelectionTests(unittest.TestCase):
    def fixture(self, root: Path):
        checkpoint = root / "focus.pt"
        checkpoint.write_bytes(b"focus")
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        sources = {"epoch105": "b32", "lung_cell0": "focus_encoder01_lung_cells_tta_xy"}
        folds = []
        for fold in range(5):
            folds.append(
                {
                    "fold": fold,
                    "routes": {
                        "b32": {"family": "candidate_refiner"},
                        "focus_encoder01_lung_cells_tta_xy": {
                            "family": "gf_focus",
                            "checkpoint": str(checkpoint),
                            "checkpoint_sha256": digest,
                            "tta_axes": "xy",
                            "tta_views": 3,
                            "input_mode": "soft_lung_seed",
                        },
                    },
                }
            )
        return {
            "decision": "QUALIFY_ENCODER_COMPOSITE_FOCUS_B32_TTA_FULL_VOLUME",
            "folds": 5,
            "focus_input_mode": "soft_lung_seed",
            "source_to_route": sources,
            "source_segmentation_thresholds": {"epoch105": 0.9, "lung_cell0": 0.25},
            "fold_records": folds,
            "public_read": False,
            "hidden_read": False,
            "submission_created": False,
        }

    def test_emits_complete_hash_validated_deployment_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = self.fixture(Path(directory))
            rows = select(payload, {"epoch105", "lung_cell0"})
            self.assertEqual(rows[0], ("INPUT", "-", "-", "soft_lung_seed"))
            self.assertEqual(sum(row[0] == "B32" for row in rows), 5)
            self.assertEqual(sum(row[0] == "FOCUS" for row in rows), 5)
            self.assertIn(
                ("TTA", "-", "focus_encoder01_lung_cells_tta_xy", "xy"), rows
            )
            self.assertIn(("SOURCE", "-", "epoch105", "b32"), rows)
            self.assertIn(("THRESHOLD", "-", "lung_cell0", "0.25"), rows)

    def test_rejects_changed_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = self.fixture(Path(directory))
            checkpoint = Path(
                payload["fold_records"][0]["routes"]
                ["focus_encoder01_lung_cells_tta_xy"]["checkpoint"]
            )
            checkpoint.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                select(payload, {"epoch105", "lung_cell0"})

    def test_rejects_source_or_fold_coverage_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = self.fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "source inventory"):
                select(payload, {"epoch105"})
            payload["fold_records"].pop()
            with self.assertRaisesRegex(ValueError, "fold coverage"):
                select(payload, {"epoch105", "lung_cell0"})


if __name__ == "__main__":
    unittest.main()
