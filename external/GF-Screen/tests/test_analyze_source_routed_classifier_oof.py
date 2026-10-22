import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from analyze_source_routed_classifier_oof import main, source_nonregression
from train_candidate_refiner import grouped_fold_rows


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AnalyzeSourceRoutedClassifierOOFTests(unittest.TestCase):
    def test_source_nonregression_protects_ap_and_f0p5(self):
        baseline = {"average_precision": 0.70, "f0p5": 0.65}
        self.assertTrue(
            source_nonregression(
                baseline, {"average_precision": 0.705, "f0p5": 0.646}
            )
        )
        self.assertFalse(
            source_nonregression(
                baseline, {"average_precision": 0.72, "f0p5": 0.64}
            )
        )
        self.assertFalse(
            source_nonregression(
                baseline, {"average_precision": 0.69, "f0p5": 0.66}
            )
        )

    def test_routes_epochs_by_source_with_exact_grouped_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = root / "index.csv"
            audit_root = root / "audits"
            audit_root.mkdir()
            sources = ("epoch105", "autopet_weak", "lung_cell0")
            rows = []
            for case_index in range(12):
                for label in (0, 1):
                    rows.append(
                        {
                            "patch_id": f"case{case_index}_p{label}",
                            "case": f"case{case_index}",
                            "cohort": f"cohort{case_index % 3}",
                            "label": str(label),
                            "source_name": sources[case_index % 3],
                        }
                    )
            with index.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            index_digest = sha256(index)
            runs = []
            for fold in range(2):
                run = root / f"run{fold}"
                run.mkdir()
                runs.append(run)
                _, held_out = grouped_fold_rows(rows, 2, fold, 17)
                held_out = sorted(held_out, key=lambda row: row["patch_id"])
                ids = np.asarray([row["patch_id"] for row in held_out])
                labels = np.asarray([int(row["label"]) for row in held_out], dtype=np.uint8)
                np.savez_compressed(
                    run / "validation_predictions_best_classification.npz",
                    patch_id=ids,
                    label=labels,
                    classification_probability=np.full(len(labels), 0.5, dtype=np.float32),
                )
                (run / "checkpoint_best_classification.pt").write_bytes(b"baseline")
                for epoch in (1, 2):
                    probabilities = np.full(len(labels), 0.5, dtype=np.float32)
                    for index_position, row in enumerate(held_out):
                        mature = row["source_name"] == "epoch105"
                        selected = mature if epoch == 1 else not mature
                        if selected:
                            probabilities[index_position] = 0.9 if labels[index_position] else 0.1
                    prediction = run / f"validation_predictions_epoch_{epoch:04d}.npz"
                    checkpoint = run / f"checkpoint_epoch_{epoch:04d}.pt"
                    np.savez_compressed(
                        prediction,
                        patch_id=ids,
                        label=labels,
                        classification_probability=probabilities,
                    )
                    checkpoint.write_bytes(f"fold{fold}-epoch{epoch}".encode())
                    high, low = ({"average_precision": 1.0, "f0p5": 1.0, "threshold": 0.5}, {"average_precision": 0.5, "f0p5": 0.5, "threshold": 0.5})
                    groups = {
                        "mature": high if epoch == 1 else low,
                        "weak_sources": high if epoch == 2 else low,
                        "lung_cells": high if epoch == 2 else low,
                    }
                    audit = {
                        "fold": fold,
                        "folds": 2,
                        "seed": 17,
                        "epoch": epoch,
                        "index_sha256": index_digest,
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": sha256(checkpoint),
                        "prediction": str(prediction),
                        "prediction_sha256": sha256(prediction),
                        "source_classification_groups": groups,
                    }
                    (audit_root / f"fold{fold}_source_epoch{epoch}.json").write_text(
                        json.dumps(audit), encoding="utf-8"
                    )
            output = root / "decision.json"
            predictions = root / "predictions.npz"
            argv = [
                "source-route",
                "--index",
                str(index),
                "--audit-root",
                str(audit_root),
                "--folds",
                "2",
                "--seed",
                "17",
                "--bootstrap",
                "0",
                "--predictions-output",
                str(predictions),
                "--output",
                str(output),
            ]
            for run in runs:
                argv.extend(("--run", str(run)))
            with patch.object(sys, "argv", argv):
                main()
            payload = json.loads(output.read_text())
            self.assertEqual(payload["decision"], "QUALIFY_SOURCE_ROUTED_CLASSIFIER")
            implementation = Path(payload["implementation"])
            self.assertTrue(implementation.is_file())
            self.assertEqual(payload["implementation_sha256"], sha256(implementation))
            self.assertGreater(payload["average_precision_gain"], 0.1)
            for selection in payload["selections"]:
                self.assertEqual(selection["routes"]["mature"]["epoch"], 1)
                self.assertEqual(selection["routes"]["weak_sources"]["epoch"], 2)
                self.assertEqual(selection["routes"]["lung_cells"]["epoch"], 2)
            with np.load(predictions, allow_pickle=False) as archive:
                self.assertEqual(len(archive["patch_id"]), len(rows))
                self.assertEqual(len(set(map(str, archive["patch_id"]))), len(rows))


if __name__ == "__main__":
    unittest.main()
