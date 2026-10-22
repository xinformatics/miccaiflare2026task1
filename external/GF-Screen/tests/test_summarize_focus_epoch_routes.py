import csv
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import summarize_focus_epoch_routes as focus_routes
from summarize_focus_epoch_routes import summarize
from train_candidate_refiner import grouped_fold_rows


class FocusEpochRouteSummaryTests(unittest.TestCase):
    def test_route_summarizer_memory_maps_large_checkpoints(self):
        source = Path(focus_routes.__file__).read_text(encoding="utf-8")
        self.assertIn("mmap=True", source)

    def build_fixture(self, root: Path):
        rows = []
        for case_index in range(12):
            for label in (0, 1):
                rows.append(
                    {
                        "patch_id": f"case{case_index}_p{label}",
                        "case": f"case{case_index}",
                        "cohort": f"cohort{case_index % 2}",
                        "label": str(label),
                        "source_name": "epoch105",
                    }
                )
        # Assign all three semantic routes within each deterministic held-out
        # fold without changing case identity or fold membership.
        for fold in range(2):
            _, held_out = grouped_fold_rows(rows, 2, fold, 17)
            cases = sorted({row["case"] for row in held_out})
            for position, case in enumerate(cases):
                source = ("epoch105", "autopet_weak", "lung_cell0")[position % 3]
                for row in rows:
                    if row["case"] == case:
                        row["source_name"] = source
        index = root / "index.csv"
        with index.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        digest = hashlib.sha256(index.read_bytes()).hexdigest()
        runs = []
        source_by_id = {row["patch_id"]: row["source_name"] for row in rows}
        for fold in range(2):
            run = root / f"fold{fold}"
            run.mkdir()
            _, held_out = grouped_fold_rows(rows, 2, fold, 17)
            positive = [row["patch_id"] for row in held_out if row["label"] == "1"]
            negative = [row["patch_id"] for row in held_out if row["label"] == "0"]
            for epoch in range(2):
                values = []
                for patch_id in positive:
                    source = source_by_id[patch_id]
                    if source == "lung_cell0":
                        score = 0.30 if epoch == 0 else 0.95
                    elif source == "autopet_weak":
                        score = 0.80 if epoch == 0 else 0.50
                    else:
                        score = 0.90 if epoch == 0 else 0.40
                    values.append([score - 0.05, score])
                config = {
                    "seed": 17,
                    "epochs": 1,
                    "batch_size": 8,
                    "lr": 3e-5,
                    "weight_decay": 1e-5,
                    "loss_recipe": "release_dice_ce",
                    "sampling_temperature": 0.5,
                    "samples_per_epoch": 128,
                    "freeze_encoder": True,
                    "input_mode": "soft_lung_seed",
                    "segmentation_threshold_max": 0.6,
                }
                torch.save(
                    {
                        "epoch": epoch,
                        "fold": fold,
                        "folds": 2,
                        "config": config,
                        "index_sha256": digest,
                    },
                    run / f"checkpoint_epoch_{epoch:04d}.pt",
                )
                np.savez_compressed(
                    run / f"validation_predictions_epoch_{epoch:04d}.npz",
                    positive_patch_id=np.asarray(positive),
                    segmentation_thresholds=np.asarray([0.4, 0.6]),
                    segmentation_dice_grid=np.asarray(values),
                    negative_patch_id=np.asarray(negative),
                    negative_nonempty_grid=np.zeros((len(negative), 2), dtype=np.uint8),
                    negative_nonempty_by_threshold=np.zeros(2, dtype=np.int64),
                )
            runs.append(run)
        return index, runs

    def test_selects_family_epochs_and_pools_thresholds(self):
        with tempfile.TemporaryDirectory() as temporary:
            index, runs = self.build_fixture(Path(temporary))
            result = summarize(index, runs, 2, 17, 2)
            self.assertEqual(
                result["source_to_route"],
                {
                    "autopet_weak": "weak_sources",
                    "epoch105": "mature",
                    "lung_cell0": "lung_cells",
                },
            )
            for fold in result["fold_records"]:
                self.assertEqual(fold["selected"]["mature"]["epoch"], 0)
                self.assertEqual(fold["selected"]["weak_sources"]["epoch"], 0)
                self.assertEqual(fold["selected"]["lung_cells"]["epoch"], 1)
            self.assertGreater(result["segmentation"]["dice"], 0.85)
            self.assertEqual(
                set(result["source_segmentation_groups"]),
                {"mature", "weak_sources", "lung_cells"},
            )

    def test_incomplete_epoch_artifacts_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            index, runs = self.build_fixture(Path(temporary))
            (runs[1] / "validation_predictions_epoch_0001.npz").unlink()
            with self.assertRaisesRegex(ValueError, "artifact coverage mismatch"):
                summarize(index, runs, 2, 17, 2)

    def test_group_metrics_builds_patch_route_map_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            index, runs = self.build_fixture(Path(temporary))
            with index.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            _, held_out = grouped_fold_rows(rows, 2, 0, 17)
            arrays = focus_routes.load_npz(
                runs[0] / "validation_predictions_epoch_0000.npz"
            )
            original = focus_routes.route_group_by_patch
            with patch.object(
                focus_routes, "route_group_by_patch", wraps=original
            ) as mapper:
                focus_routes.group_metrics(arrays, held_out, "mature")
            mapper.assert_called_once_with(held_out)


if __name__ == "__main__":
    unittest.main()
