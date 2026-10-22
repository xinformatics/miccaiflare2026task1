import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from candidate_refiner_model import CandidateRefiner
from evaluate_candidate_refiner_transfer import checkpoint_spec, evaluate_transfer, sha256
from train_candidate_refiner import grouped_fold_rows


class CandidateRefinerTransferEvaluationTests(unittest.TestCase):
    def test_checkpoint_spec_rejects_architecture_drift(self):
        model = CandidateRefiner(base_channels=4)
        checkpoint = {
            "state_dict": model.state_dict(),
            "fold": 0,
            "folds": 2,
            "config": {"seed": 7, "base_channels": 8},
        }
        with self.assertRaisesRegex(ValueError, "architecture metadata"):
            checkpoint_spec(checkpoint)

    def test_checkpoint_spec_binds_seed_support_prompt_channel(self):
        model = CandidateRefiner(
            base_channels=4, prompt_masked_classifier=True, prompt_channel=1
        )
        checkpoint = {
            "state_dict": model.state_dict(),
            "fold": 0,
            "folds": 2,
            "config": {
                "seed": 7,
                "base_channels": 4,
                "prompt_masked_classifier": True,
                "prompt_channel": 1,
            },
        }
        with self.assertRaisesRegex(ValueError, "fixed .* input layout"):
            checkpoint_spec(checkpoint)

    def test_cross_index_requires_explicit_transfer_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            cases = ["a0", "a1", "b0", "b1"]
            for case in cases:
                rows.append({"patch_id": case, "case": case, "cohort": case[0]})
            _, validation = grouped_fold_rows(rows, 2, 0, 11)
            validation_cases = {row["case"] for row in validation}
            for row in rows:
                path = root / f"{row['case']}.npz"
                label = float(row["case"] == sorted(validation_cases)[0])
                target = np.zeros((1, 8, 8, 8), dtype=np.float32)
                if label:
                    target[:, 3:5, 3:5, 3:5] = 1
                np.savez_compressed(
                    path,
                    image=np.zeros((3, 8, 8, 8), dtype=np.float32),
                    target=target,
                    label=np.asarray(label, dtype=np.float32),
                )
                row.update({"path": str(path), "label": str(int(label))})
            index = root / "index.csv"
            with index.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            model = CandidateRefiner(base_channels=4)
            checkpoint_path = root / "checkpoint.pt"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": 3,
                    "fold": 0,
                    "folds": 2,
                    "config": {
                        "seed": 11,
                        "base_channels": 4,
                        "segmentation_threshold_max": 0.95,
                    },
                    "index_sha256": "0" * 64,
                },
                checkpoint_path,
            )
            with self.assertRaisesRegex(ValueError, "allow-index-transfer"):
                evaluate_transfer(
                    index, checkpoint_path, root / "blocked.json",
                    root / "blocked.npz", 2, 0, False, torch.device("cpu")
                )
            output = root / "result.json"
            predictions = root / "predictions.npz"
            result = evaluate_transfer(
                index, checkpoint_path, output, predictions, 2, 0, True,
                torch.device("cpu")
            )
            self.assertTrue(result["cross_index_transfer"])
            self.assertEqual(result["evaluation_index_sha256"], sha256(index))
            self.assertEqual(result["validation_cases"], 2)
            self.assertEqual(result["validation_patches"], 2)
            self.assertTrue(output.is_file())
            self.assertTrue(predictions.is_file())
            self.assertEqual(result["predictions_sha256"], sha256(predictions))
            self.assertEqual(result["segmentation_threshold_max"], 0.95)
            with np.load(predictions, allow_pickle=False) as archive:
                self.assertEqual(archive["segmentation_dice_grid"].shape[1], 16)
                self.assertAlmostEqual(
                    float(archive["segmentation_thresholds"][-1]), 0.95
                )
            self.assertFalse(result["public_read"])
            self.assertFalse(result["hidden_read"])


if __name__ == "__main__":
    unittest.main()
