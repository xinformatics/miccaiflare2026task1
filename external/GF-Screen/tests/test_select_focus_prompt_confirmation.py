import unittest
import csv
import hashlib
import tempfile
from pathlib import Path

import numpy as np
import torch

from select_focus_prompt_confirmation import decide, summarize
from train_candidate_refiner import grouped_fold_rows


class PromptFocusConfirmationTests(unittest.TestCase):
    def plan(self):
        return {
            "fold0_gate": {
                "case_balanced_segmentation_dice_min": 0.55,
                "negative_nonempty_rate_max": 0.93,
                "selection_score_min": 0.531,
            },
            "future_fold1_gate_if_fold0_passes": {
                "case_balanced_segmentation_dice_min": 0.591,
                "two_fold_mean_gain_over_base32_min": 0.02,
                "negative_nonempty_rate_max": 0.95,
            },
        }

    def records(self):
        return [
            {
                "fold": 0,
                "focus_case_balanced_dice": 0.56,
                "dice_gain": 0.057,
                "negative_nonempty_rate": 0.90,
                "selection_score": 0.542,
            },
            {
                "fold": 1,
                "focus_case_balanced_dice": 0.60,
                "dice_gain": 0.024,
                "negative_nonempty_rate": 0.91,
                "selection_score": 0.582,
            },
        ]

    def test_all_gates_expand(self):
        result = decide(self.records(), self.plan())
        self.assertTrue(result["passed"])
        self.assertEqual(result["decision"], "EXPAND_FIVE_FOLD")

    def test_fold1_regression_rejects(self):
        records = self.records()
        records[1]["focus_case_balanced_dice"] = 0.58
        records[1]["dice_gain"] = 0.004
        result = decide(records, self.plan())
        self.assertFalse(result["passed"])
        self.assertFalse(result["fold1_gate_passed"])

    def test_mean_gain_must_pass(self):
        records = self.records()
        records[0]["dice_gain"] = 0.016
        records[1]["dice_gain"] = 0.021
        result = decide(records, self.plan())
        self.assertFalse(result["passed"])
        self.assertFalse(result["two_fold_mean_gate_passed"])

    def test_file_level_summary_checks_exact_fold_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "index.csv"
            rows = [
                {
                    "patch_id": f"p{i}",
                    "case": f"case{i}",
                    "cohort": "cohort",
                    "label": "1",
                }
                for i in range(10)
            ]
            with index.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            plan = self.plan()
            plan.update(
                {
                    "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
                    "pretrained_sha256": "pretrained",
                    "training": {
                        "epochs": 10,
                        "batch_size": 8,
                        "workers": 2,
                        "learning_rate": 3e-5,
                        "weight_decay": 1e-5,
                        "loss_recipe": "release_dice_ce",
                        "sampling_temperature": 0.5,
                        "samples_per_epoch": 2048,
                        "folds": 5,
                        "seed": 20260823,
                    },
                }
            )
            runs, small_paths = [], []
            for fold, focus_dice, small_dice, score in (
                (0, 0.56, 0.50, 0.542),
                (1, 0.60, 0.575, 0.582),
            ):
                _, validation = grouped_fold_rows(rows, 5, fold, 20260823)
                ids = np.asarray([row["patch_id"] for row in validation])
                run = root / f"run{fold}"
                run.mkdir()
                config = {
                    "epochs": 10,
                    "batch_size": 8,
                    "workers": 2,
                    "lr": 3e-5,
                    "weight_decay": 1e-5,
                    "loss_recipe": "release_dice_ce",
                    "sampling_temperature": 0.5,
                    "samples_per_epoch": 2048,
                    "freeze_encoder": True,
                    "input_mode": "soft_seed",
                    "folds": 5,
                    "seed": 20260823,
                }
                torch.save(
                    {
                        "fold": fold,
                        "epoch": 8,
                        "index_sha256": plan["index_sha256"],
                        "pretrained_sha256": "pretrained",
                        "config": config,
                        "metrics": {
                            "segmentation_dice": focus_dice,
                            "negative_nonempty_rate": 0.90,
                            "selection_score": score,
                        },
                    },
                    run / "checkpoint_best.pt",
                )
                np.savez_compressed(
                    run / "validation_predictions_best.npz",
                    positive_patch_id=ids,
                    segmentation_dice_grid=np.full((len(ids), 1), focus_dice),
                    segmentation_thresholds=np.asarray([0.5]),
                )
                small_path = root / f"small{fold}.npz"
                np.savez_compressed(
                    small_path,
                    positive_patch_id=ids,
                    segmentation_dice_grid=np.full((len(ids), 1), small_dice),
                    segmentation_thresholds=np.asarray([0.5]),
                )
                runs.append(run)
                small_paths.append(small_path)
            result = summarize(plan, index, runs, small_paths)
            self.assertEqual(result["decision"], "EXPAND_FIVE_FOLD")
            self.assertEqual([row["fold"] for row in result["fold_records"]], [0, 1])


if __name__ == "__main__":
    unittest.main()
