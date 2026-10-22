import hashlib
import tempfile
import unittest
from pathlib import Path

from compose_focus_encoder_upgrade_routes import compose


class ComposeFocusEncoderUpgradeRouteTests(unittest.TestCase):
    def fixture(self, root: Path):
        candidate_checkpoint = root / "candidate.pt"
        frozen_checkpoint = root / "frozen.pt"
        candidate_checkpoint.write_bytes(b"candidate")
        frozen_checkpoint.write_bytes(b"frozen")
        candidate_hash = hashlib.sha256(candidate_checkpoint.read_bytes()).hexdigest()
        frozen_hash = hashlib.sha256(frozen_checkpoint.read_bytes()).hexdigest()
        sources = {
            "epoch105": "mature",
            "autopet_weak": "weak_sources",
            "lung_cell0": "lung_cells",
        }
        candidate_records = []
        base_records = []
        for fold in range(5):
            selected = {
                group: {
                    "epoch": 4,
                    "checkpoint": str(candidate_checkpoint),
                    "checkpoint_sha256": candidate_hash,
                }
                for group in ("mature", "weak_sources", "lung_cells")
            }
            candidate_records.append({"fold": fold, "selected": selected})
            base_records.append(
                {
                    "fold": fold,
                    "routes": {
                        "b32": {"family": "candidate_refiner"},
                        "focus_weak_sources_tta_xy": {
                            "family": "gf_focus",
                            "group": "weak_sources",
                            "checkpoint": str(frozen_checkpoint),
                            "checkpoint_sha256": frozen_hash,
                        },
                        "focus_lung_cells_tta_xy": {
                            "family": "gf_focus",
                            "group": "lung_cells",
                            "checkpoint": str(frozen_checkpoint),
                            "checkpoint_sha256": frozen_hash,
                        },
                    },
                }
            )
        candidate = {
            "index_sha256": "index",
            "folds": 5,
            "training_recipe": {
                "freeze_encoder": False,
                "encoder_lr_scale": 0.1,
                "input_mode": "soft_lung_seed",
            },
            "source_to_route": sources,
            "source_segmentation_groups": {
                "mature": {"threshold": 0.8},
                "weak_sources": {"threshold": 0.4},
                "lung_cells": {"threshold": 0.3},
            },
            "fold_records": candidate_records,
            "public_read": False,
            "hidden_read": False,
            "submission_created": False,
        }
        base = {
            "decision": "QUALIFY_COMPOSITE_FOCUS_B32_TTA_FULL_VOLUME",
            "index_sha256": "index",
            "folds": 5,
            "b32_groups": ["mature"],
            "source_to_route": {
                "epoch105": "b32",
                "autopet_weak": "focus_weak_sources_tta_xy",
                "lung_cell0": "focus_lung_cells_tta_xy",
            },
            "source_segmentation_thresholds": {
                "epoch105": 0.9,
                "autopet_weak": 0.35,
                "lung_cell0": 0.25,
            },
            "fold_records": base_records,
            "public_read": False,
            "hidden_read": False,
            "submission_created": False,
        }
        upgrade = {
            "decision": "PROMOTE_ENCODER_UPGRADE_TO_ROUTED_FULL_VOLUME_OOF",
            "passed": True,
            "qualified_routes": ["weak_sources"],
            "mature_route": "B32_UNCHANGED",
            "public_read": False,
            "hidden_read": False,
            "submission_created": False,
        }
        tta = {
            "decision": "QUALIFY_ROUTED_TTA_FULL_VOLUME",
            "qualified_groups": ["weak_sources"],
            "groups": {"weak_sources": {"tta_xy": {"threshold": 0.45}}},
            "public_read": False,
            "hidden_read": False,
            "submission_created": False,
        }
        return base, candidate, upgrade, tta

    def test_replaces_only_qualified_route_and_preserves_b32_and_frozen_focus(self):
        with tempfile.TemporaryDirectory() as directory:
            base, candidate, upgrade, tta = self.fixture(Path(directory))
            result = compose(base, candidate, upgrade, tta)
            self.assertEqual(
                result["decision"],
                "QUALIFY_ENCODER_COMPOSITE_FOCUS_B32_TTA_FULL_VOLUME",
            )
            self.assertEqual(result["source_to_route"]["epoch105"], "b32")
            self.assertEqual(
                result["source_to_route"]["lung_cell0"],
                "focus_lung_cells_tta_xy",
            )
            self.assertEqual(
                result["source_to_route"]["autopet_weak"],
                "focus_encoder01_weak_sources_tta_xy",
            )
            self.assertEqual(
                result["source_segmentation_thresholds"]["autopet_weak"], 0.45
            )
            self.assertEqual(
                set(result["fold_records"][0]["routes"]),
                {
                    "b32",
                    "focus_lung_cells_tta_xy",
                    "focus_encoder01_weak_sources_tta_xy",
                },
            )

    def test_uses_candidate_notta_when_tta_does_not_requalify(self):
        with tempfile.TemporaryDirectory() as directory:
            base, candidate, upgrade, tta = self.fixture(Path(directory))
            tta["decision"] = "KEEP_NOTTA"
            tta["qualified_groups"] = []
            result = compose(base, candidate, upgrade, tta)
            self.assertEqual(
                result["source_to_route"]["autopet_weak"],
                "focus_encoder01_weak_sources_notta",
            )
            self.assertEqual(
                result["source_segmentation_thresholds"]["autopet_weak"], 0.4
            )
            # The retained frozen lung route still contributes qualified TTA.
            self.assertEqual(
                result["decision"],
                "QUALIFY_ENCODER_COMPOSITE_FOCUS_B32_TTA_FULL_VOLUME",
            )

    def test_never_allows_mature_encoder_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            base, candidate, upgrade, tta = self.fixture(Path(directory))
            upgrade["qualified_routes"] = ["mature"]
            with self.assertRaisesRegex(ValueError, "did not qualify"):
                compose(base, candidate, upgrade, tta)


if __name__ == "__main__":
    unittest.main()
