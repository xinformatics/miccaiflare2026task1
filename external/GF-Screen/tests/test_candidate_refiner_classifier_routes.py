import hashlib
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from build_source_routed_classifier_patch_oof import pooled_threshold
from candidate_refiner_classifier_route_selection import GROUP_SOURCES, route_records


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CandidateRefinerClassifierRouteTests(unittest.TestCase):
    def test_patch_oof_builder_preserves_contours_and_exact707_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = root / "index.csv"
            sources = [
                "epoch105", "currentgf", "autopet_weak", "lndb_weak",
                *(f"lung_cell{value}" for value in range(8)),
            ]
            rows = []
            for value in range(707):
                rows.append({
                    "patch_id": f"patch{value:04d}",
                    "case": f"case{value:04d}",
                    "label": str((value // len(sources)) % 2),
                    "source_name": sources[value % len(sources)],
                })
            with index.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            labels = np.asarray([int(row["label"]) for row in rows], dtype=np.uint8)
            probabilities = np.where(labels == 1, 0.8, 0.2).astype(np.float32)
            prediction = root / "classification_oof.npz"
            np.savez_compressed(
                prediction,
                patch_id=np.asarray([row["patch_id"] for row in rows]),
                label=labels,
                classification_probability=probabilities,
                fold=np.asarray([value % 5 for value in range(707)], dtype=np.uint8),
            )
            decision = root / "DECISION.json"
            decision.write_text(json.dumps({
                "decision": "QUALIFY_SOURCE_ROUTED_CLASSIFIER",
                "index_sha256": digest(index),
                "predictions": str(prediction),
                "predictions_sha256": digest(prediction),
                "average_precision_gain_95ci": [0.01, 0.02],
            }))
            baseline = root / "baseline.json"
            baseline.write_text(json.dumps({
                "index_sha256": digest(index),
                "coverage": {"cases": 707, "patches": 707, "every_patch_exactly_once": True},
                "classification": {"average_precision": 0.5},
                "segmentation": {"dice": 0.321, "threshold": 0.9},
                "source_segmentation": {source: {"dice": 0.1} for source in sources},
                "source_segmentation_groups": {},
            }))
            output = root / "summary.json"
            script = Path(__file__).resolve().parents[1] / "build_source_routed_classifier_patch_oof.py"
            subprocess.run(
                [sys.executable, str(script), "--index", str(index),
                 "--decision", str(decision), "--baseline", str(baseline),
                 "--output", str(output)],
                check=True, capture_output=True, text=True,
            )
            result = json.loads(output.read_text())
            self.assertEqual(result["coverage"]["cases"], 707)
            self.assertEqual(result["segmentation"]["dice"], 0.321)
            self.assertEqual(result["classification"]["average_precision"], 1.0)
            self.assertEqual(set(result["source_classification"]), set(sources))
            self.assertEqual(set(result["source_classification_groups"]), set(GROUP_SOURCES))
            self.assertEqual(
                result["checkpoint_selection"]["selected_strategy"],
                "classifier_routed",
            )

    def test_route_records_require_exact_folds_groups_and_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = root / "route_index.csv"
            index.write_text("patch_id,case\np,c\n")
            predictions = root / "route_predictions.npz"
            predictions.write_bytes(b"cross-fitted probabilities")
            selections = []
            for fold in range(5):
                routes = {}
                for group in GROUP_SOURCES:
                    checkpoint = root / f"fold{fold}_{group}.pt"
                    checkpoint.write_bytes(f"{fold}:{group}".encode())
                    routes[group] = {
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": digest(checkpoint),
                    }
                selections.append({"fold": fold, "routes": routes})
            payload = {
                "decision": "QUALIFY_SOURCE_ROUTED_CLASSIFIER",
                "selection_basis": "source-family checkpoint routing on exact707 grouped held-out patches",
                "index": str(index),
                "index_sha256": digest(index),
                "predictions": str(predictions),
                "predictions_sha256": digest(predictions),
                "checks": {
                    "average_precision_gain": True,
                    "f0p5_gain": True,
                    "source_nonregression": True,
                },
                "public_read": False,
                "hidden_read": False,
                "submission_created": False,
                "selections": selections,
            }
            records = route_records(payload)
            self.assertEqual(sum(row[0] == "ROUTE" for row in records), 15)
            self.assertEqual(sum(row[0] == "SOURCE" for row in records), 12)
            fold_two = route_records(payload, 2)
            self.assertEqual(sum(row[0] == "ROUTE" for row in fold_two), 3)
            self.assertEqual(sum(row[0] == "SOURCE" for row in fold_two), 12)

            original = payload["selections"]
            payload["selections"] = []
            with self.assertRaisesRegex(ValueError, "exact five-fold"):
                route_records(payload)
            payload["selections"] = original

            selections[2]["routes"]["mature"]["checkpoint_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                route_records(payload, 2)

    def test_route_records_reject_unqualified_or_incomplete_decision(self):
        with self.assertRaisesRegex(ValueError, "did not qualify"):
            route_records({"decision": "KEEP_GLOBAL_CLASSIFIER"})
        with self.assertRaisesRegex(ValueError, "provenance"):
            route_records({
                "decision": "QUALIFY_SOURCE_ROUTED_CLASSIFIER",
                "selection_basis": "source-family checkpoint routing on exact707 grouped held-out patches",
                "index_sha256": "a" * 64,
                "checks": {
                    "average_precision_gain": True,
                    "f0p5_gain": True,
                    "source_nonregression": True,
                },
                "public_read": False,
                "hidden_read": False,
                "submission_created": False,
                "selections": [],
            })

    def test_pooled_threshold_prefers_precision_for_equal_f0p5(self):
        labels = np.asarray([1, 1, 0, 0], dtype=np.uint8)
        probabilities = np.asarray([0.9, 0.7, 0.6, 0.1], dtype=np.float64)
        result = pooled_threshold(labels, probabilities)
        self.assertGreaterEqual(result["threshold"], 0.6)
        self.assertGreater(result["f0p5"], 0.8)


if __name__ == "__main__":
    unittest.main()
