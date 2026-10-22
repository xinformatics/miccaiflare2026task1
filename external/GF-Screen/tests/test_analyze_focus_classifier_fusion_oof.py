import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from analyze_focus_classifier_fusion_oof import main
from train_candidate_refiner import grouped_fold_rows
from train_focus_candidate_refiner import FOCUS_CONFIDENCE_FEATURE_NAMES


class AnalyzeFocusClassifierFusionOOFTests(unittest.TestCase):
    def test_cross_fits_exact_grouped_coverage_and_promotes_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "index.csv"
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
                            "source_name": sources[case_index % len(sources)],
                            "seed_mm3": "100",
                            "max_source_support": "0.5",
                            "mean_source_support": "0.25",
                        }
                    )
            with index.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            b32_paths, focus_paths = [], []
            for fold in range(2):
                _, held_out = grouped_fold_rows(rows, 2, fold, 17)
                held_out = sorted(held_out, key=lambda row: row["patch_id"])
                patch_ids = np.asarray([row["patch_id"] for row in held_out])
                labels = np.asarray([int(row["label"]) for row in held_out], dtype=np.uint8)
                b32 = root / f"b32_{fold}.npz"
                focus = root / f"focus_{fold}.npz"
                np.savez_compressed(
                    b32,
                    patch_id=patch_ids,
                    label=labels,
                    classification_probability=np.full(len(labels), 0.5, dtype=np.float32),
                )
                features = np.zeros(
                    (len(labels), len(FOCUS_CONFIDENCE_FEATURE_NAMES)),
                    dtype=np.float32,
                )
                features[:, 0] = np.where(labels == 1, 0.9, 0.1)
                np.savez_compressed(
                    focus,
                    all_patch_id=patch_ids,
                    all_label=labels,
                    confidence_feature_names=np.asarray(FOCUS_CONFIDENCE_FEATURE_NAMES),
                    confidence_features=features,
                )
                b32_paths.append(b32)
                focus_paths.append(focus)
            output = root / "decision.json"
            predictions = root / "predictions.npz"
            argv = [
                "fusion",
                "--index",
                str(index),
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
            for b32, focus in zip(b32_paths, focus_paths):
                argv.extend(("--b32-prediction", str(b32)))
                argv.extend(("--focus-prediction", str(focus)))
            with patch.object(sys, "argv", argv):
                main()
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["decision"], "QUALIFY_FOCUS_CLASSIFIER_FUSION")
            self.assertGreater(payload["average_precision_gain"], 0.1)
            self.assertEqual(set(payload["by_source"]), {"mature", "weak_sources", "lung_cells"})
            self.assertTrue(all(payload["checks"].values()))
            with np.load(predictions, allow_pickle=False) as archive:
                self.assertEqual(len(archive["patch_id"]), len(rows))
                self.assertEqual(len(set(map(str, archive["patch_id"]))), len(rows))


if __name__ == "__main__":
    unittest.main()
