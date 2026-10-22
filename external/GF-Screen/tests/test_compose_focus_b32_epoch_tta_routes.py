import hashlib
import tempfile
import unittest
from pathlib import Path

from compose_focus_b32_epoch_tta_routes import compose


SOURCES = {
    "epoch105": "mature",
    "currentgf": "mature",
    "autopet_weak": "weak_sources",
    "lndb_weak": "weak_sources",
    "lung_cell0": "lung_cells",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CompositeFocusRouteTests(unittest.TestCase):
    def payloads(self, root):
        records = []
        for fold in range(5):
            selected = {}
            for index, group in enumerate(sorted(set(SOURCES.values()))):
                checkpoint = root / f"fold{fold}_{group}.pt"
                checkpoint.write_bytes(f"{fold}:{group}".encode())
                selected[group] = {
                    "epoch": fold + index,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": digest(checkpoint),
                }
            records.append({"fold": fold, "selected": selected})
        epoch = {
            "folds": 5,
            "index_sha256": "index",
            "training_recipe": {"input_mode": "soft_lung_seed"},
            "source_to_route": SOURCES,
            "source_segmentation_groups": {
                group: {"threshold": threshold}
                for group, threshold in (
                    ("mature", 0.2),
                    ("weak_sources", 0.3),
                    ("lung_cells", 0.4),
                )
            },
            "fold_records": records,
            "public_read": False,
            "hidden_read": False,
            "submission_created": False,
        }
        source_families = {
            source: "b32" if group == "mature" else "focus"
            for source, group in SOURCES.items()
        }
        mixed = {
            "decision": "QUALIFY_MIXED_FOCUS_B32_FULL_VOLUME",
            "index_sha256": "index",
            "source_to_route": source_families,
            "source_segmentation_thresholds": {
                source: 0.5 for source in SOURCES
            },
            "groups": {
                group: {
                    "selected_family": "b32" if group == "mature" else "focus"
                }
                for group in set(SOURCES.values())
            },
            "public_read": False,
            "hidden_read": False,
            "submission_created": False,
        }
        tta = {
            "decision": "QUALIFY_ROUTED_TTA_FULL_VOLUME",
            "qualified_groups": ["weak_sources"],
            "source_to_tta": {
                source: group == "weak_sources" for source, group in SOURCES.items()
            },
            "groups": {
                "weak_sources": {"tta_xy": {"threshold": 0.35}},
                "lung_cells": {"tta_xy": {"threshold": 0.45}},
            },
            "public_read": False,
            "hidden_read": False,
            "submission_created": False,
        }
        return epoch, mixed, tta

    def test_composes_b32_epoch_routes_and_only_qualified_tta(self):
        with tempfile.TemporaryDirectory() as temporary:
            epoch, mixed, tta = self.payloads(Path(temporary))
            result = compose(epoch, mixed, tta)
        self.assertEqual(
            result["decision"], "QUALIFY_COMPOSITE_FOCUS_B32_TTA_FULL_VOLUME"
        )
        self.assertEqual(result["source_to_route"]["epoch105"], "b32")
        self.assertEqual(
            result["source_to_route"]["autopet_weak"],
            "focus_weak_sources_tta_xy",
        )
        self.assertEqual(
            result["source_to_route"]["lung_cell0"],
            "focus_lung_cells_notta",
        )
        self.assertEqual(result["source_segmentation_thresholds"]["epoch105"], 0.5)
        self.assertEqual(
            result["source_segmentation_thresholds"]["autopet_weak"], 0.35
        )
        self.assertEqual(result["source_segmentation_thresholds"]["lung_cell0"], 0.4)
        self.assertEqual(result["fold_records"][0]["routes"]["b32"]["family"], "candidate_refiner")
        self.assertEqual(
            result["fold_records"][0]["routes"]["focus_weak_sources_tta_xy"]["tta_views"],
            3,
        )

    def test_rejects_tta_flag_disagreement(self):
        with tempfile.TemporaryDirectory() as temporary:
            epoch, mixed, tta = self.payloads(Path(temporary))
            tta["source_to_tta"]["autopet_weak"] = False
            with self.assertRaisesRegex(ValueError, "disagree"):
                compose(epoch, mixed, tta)

    def test_rejects_public_tuned_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            epoch, mixed, tta = self.payloads(Path(temporary))
            tta["public_read"] = True
            with self.assertRaisesRegex(ValueError, "local-only"):
                compose(epoch, mixed, tta)


if __name__ == "__main__":
    unittest.main()
