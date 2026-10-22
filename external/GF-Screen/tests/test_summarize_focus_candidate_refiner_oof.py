import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from summarize_focus_candidate_refiner_oof import summarize
from train_candidate_refiner import grouped_fold_rows


class SummarizeFocusOOFTests(unittest.TestCase):
    def test_freezes_threshold_from_exact_grouped_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = root / "index.csv"
            rows = []
            for case_index in range(10):
                source_name = (
                    "epoch105"
                    if case_index % 3 == 2
                    else f"lung_cell{case_index % 2}"
                )
                for label in (0, 1):
                    rows.append(
                        {
                            "patch_id": f"case{case_index}_p{label}",
                            "case": f"case{case_index}",
                            "cohort": "cohort_a",
                            "label": str(label),
                            "source_name": source_name,
                        }
                    )
            with index.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            source_by_id = {row["patch_id"]: row["source_name"] for row in rows}
            digest = __import__("hashlib").sha256(index.read_bytes()).hexdigest()
            checkpoints, predictions = [], []
            for fold in range(2):
                _, held_out = grouped_fold_rows(rows, 2, fold, 11)
                positive = [row["patch_id"] for row in held_out if row["label"] == "1"]
                negative_ids = [
                    row["patch_id"] for row in held_out if row["label"] == "0"
                ]
                negative = len(negative_ids)
                checkpoint = root / f"fold{fold}.pt"
                prediction = root / f"fold{fold}.npz"
                torch.save(
                    {
                        "fold": fold,
                        "folds": 2,
                        "epoch": 3,
                        "index_sha256": digest,
                        "loss_recipe": "release_dice_ce",
                        "config": {"batch_size": 8},
                        "metrics": {"negative_patches": negative},
                    },
                    checkpoint,
                )
                np.savez_compressed(
                    prediction,
                    positive_patch_id=np.asarray(positive),
                    segmentation_thresholds=np.asarray([0.4, 0.6], dtype=np.float32),
                    segmentation_dice_grid=np.asarray(
                        [
                            [0.9, 0.2]
                            if source_by_id[patch_id] == "epoch105"
                            else [0.2, 0.9]
                            for patch_id in positive
                        ],
                        dtype=np.float32,
                    ),
                    negative_nonempty_by_threshold=np.asarray(
                        [negative, 0], dtype=np.int64
                    ),
                    negative_patch_id=np.asarray(negative_ids),
                    negative_nonempty_grid=np.asarray(
                        [[1, 0] for _ in negative_ids], dtype=np.uint8
                    ),
                )
                checkpoints.append(checkpoint); predictions.append(prediction)
            result = summarize(index, checkpoints, predictions, 2, 11)
            self.assertAlmostEqual(result["segmentation_threshold"], 0.6, places=5)
            self.assertEqual(result["positive_patches"], 10)
            self.assertEqual(result["negative_patches"], 10)
            self.assertEqual(result["loss_recipe"], "release_dice_ce")
            self.assertEqual(result["batch_size"], 8)
            self.assertEqual(result["input_mode"], "soft")
            self.assertEqual(result["training_recipe"]["encoder_lr_scale"], 1.0)
            self.assertFalse(result["training_recipe"]["freeze_encoder"])
            self.assertEqual(
                set(result["source_segmentation"]),
                {"epoch105", "lung_cell0", "lung_cell1"},
            )
            self.assertAlmostEqual(
                result["source_segmentation"]["epoch105"]["threshold"], 0.4
            )
            self.assertAlmostEqual(
                result["source_segmentation"]["lung_cell0"]["threshold"], 0.6
            )
            self.assertEqual(
                result["source_segmentation"]["lung_cell0"]["source_group"],
                "lung_cells",
            )
            self.assertEqual(
                result["source_segmentation"]["lung_cell1"],
                result["source_segmentation"]["lung_cell0"],
            )
            self.assertEqual(
                set(result["source_segmentation_groups"]), {"mature", "lung_cells"}
            )
            self.assertEqual(
                result["source_segmentation_groups"]["mature"]["sources"],
                ["epoch105"],
            )
            self.assertEqual(
                result["source_segmentation_groups"]["lung_cells"]["sources"],
                ["lung_cell0", "lung_cell1"],
            )
            self.assertEqual(
                result["source_segmentation_groups"]["mature"]["negative_nonempty_rate"],
                1.0,
            )
            self.assertEqual(
                result["source_segmentation_groups"]["lung_cells"]["negative_nonempty_rate"],
                0.0,
            )
            changed = torch.load(checkpoints[1], map_location="cpu", weights_only=False)
            changed["config"]["input_mode"] = "soft_seed"
            torch.save(changed, checkpoints[1])
            with self.assertRaisesRegex(ValueError, "different input modes"):
                summarize(index, checkpoints, predictions, 2, 11)


if __name__ == "__main__":
    unittest.main()
