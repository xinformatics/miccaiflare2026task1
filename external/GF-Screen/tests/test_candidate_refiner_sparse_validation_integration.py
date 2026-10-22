import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train_candidate_refiner.py"


class SparseValidationIntegrationTest(unittest.TestCase):
    def test_nonvalidation_epoch_keeps_training_while_history_stays_scored(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for case_index in range(8):
                for label in (0, 1):
                    patch_id = f"case{case_index}_label{label}"
                    path = root / f"{patch_id}.npz"
                    image = np.zeros((3, 8, 8, 8), dtype=np.float32)
                    image[0] = 0.25 + 0.05 * case_index
                    image[1, 2:6, 2:6, 2:6] = 1.0
                    target = np.zeros((1, 8, 8, 8), dtype=np.uint8)
                    if label:
                        target[:, 3:5, 3:5, 3:5] = 1
                    np.savez_compressed(path, image=image, target=target, label=label)
                    rows.append(
                        {
                            "patch_id": patch_id,
                            "path": str(path),
                            "case": f"case{case_index}",
                            "cohort": "synthetic",
                            "label": str(label),
                            "source_name": "epoch105",
                        }
                    )
            index = root / "index.csv"
            with index.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            output = root / "run"
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = ""
            environment["PYTHONPATH"] = str(ROOT)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(TRAIN),
                    "--index",
                    str(index),
                    "--output",
                    str(output),
                    "--folds",
                    "2",
                    "--fold",
                    "0",
                    "--epochs",
                    "3",
                    "--batch-size",
                    "2",
                    "--workers",
                    "0",
                    "--base-channels",
                    "4",
                    "--samples-per-epoch",
                    "2",
                    "--validation-interval",
                    "3",
                    "--validation-batch-size",
                    "4",
                    "--seed",
                    "17",
                ],
                cwd=ROOT,
                env=environment,
                text=True,
                capture_output=True,
                timeout=120,
                check=True,
            )
            trace = [
                json.loads(line)
                for line in completed.stdout.splitlines()
                if line.startswith("{")
            ]
            self.assertEqual([row["epoch"] for row in trace], [1, 2, 3])
            self.assertEqual([row["validated"] for row in trace], [True, False, True])
            history = json.loads((output / "history.json").read_text(encoding="utf-8"))
            self.assertEqual([row["epoch"] for row in history], [1, 3])
            checkpoint = torch.load(
                output / "checkpoint_last.pt", map_location="cpu", weights_only=False
            )
            self.assertEqual(checkpoint["epoch"], 3)
            self.assertEqual(checkpoint["config"]["validation_interval"], 3)
            self.assertEqual(checkpoint["config"]["validation_batch_size"], 4)
            self.assertEqual([row["epoch"] for row in checkpoint["history"]], [1, 3])
            self.assertTrue((output / "checkpoint_best.pt").is_file())
            self.assertTrue((output / "COMPLETE.txt").is_file())


if __name__ == "__main__":
    unittest.main()
