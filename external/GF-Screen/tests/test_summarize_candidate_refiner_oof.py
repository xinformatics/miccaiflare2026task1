import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from summarize_candidate_refiner_oof import (
    case_balanced_threshold_metrics,
    cohort_metrics,
    deployment_classification_thresholds,
    deployment_segmentation_thresholds,
    file_sha256,
    main,
    pool_source_classification_groups,
    pool_source_segmentation_groups,
    source_classification_metrics,
    source_segmentation_metrics,
    validate_composition_result,
    validate_transfer_result,
)


class CandidateRefinerOOFTests(unittest.TestCase):
    def test_composition_audit_binds_both_checkpoints_and_prediction_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            class_checkpoint_path = root / "class.pt"
            contour_checkpoint_path = root / "contour.pt"
            class_predictions = root / "class.npz"
            contour_predictions = root / "contour.npz"
            output = root / "composed.npz"
            audit = root / "composition.json"
            for path, value in (
                (class_checkpoint_path, b"class-checkpoint"),
                (contour_checkpoint_path, b"contour-checkpoint"),
                (class_predictions, b"class-predictions"),
                (contour_predictions, b"contour-predictions"),
                (output, b"composed"),
            ):
                path.write_bytes(value)
            index_digest = "a" * 64
            class_checkpoint = {
                "fold": 2,
                "folds": 5,
                "epoch": 9,
                "config": {"seed": 17},
                "index_sha256": index_digest,
            }
            contour_checkpoint = {
                **class_checkpoint,
                "epoch": 7,
            }

            def identity(path, checkpoint):
                return {
                    "path": str(path.resolve()),
                    "sha256": file_sha256(path),
                    "fold": checkpoint["fold"],
                    "folds": checkpoint["folds"],
                    "seed": checkpoint["config"]["seed"],
                    "epoch": checkpoint["epoch"],
                    "index_sha256": checkpoint["index_sha256"],
                }

            payload = {
                "schema_version": 1,
                "classification": str(class_predictions.resolve()),
                "classification_sha256": file_sha256(class_predictions),
                "classification_checkpoint": identity(
                    class_checkpoint_path, class_checkpoint
                ),
                "contour": str(contour_predictions.resolve()),
                "contour_sha256": file_sha256(contour_predictions),
                "contour_checkpoint": identity(
                    contour_checkpoint_path, contour_checkpoint
                ),
                "output": str(output.resolve()),
                "output_sha256": file_sha256(output),
                "labels_read": False,
                "public_read": False,
                "hidden_read": False,
            }
            audit.write_text(json.dumps(payload), encoding="utf-8")
            validated = validate_composition_result(
                audit,
                index_digest,
                class_checkpoint_path,
                class_checkpoint,
                contour_checkpoint_path,
                contour_checkpoint,
                output,
            )
            self.assertEqual(validated["output_sha256"], file_sha256(output))
            payload["contour_checkpoint"]["epoch"] = 8
            audit.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contour_checkpoint"):
                validate_composition_result(
                    audit,
                    index_digest,
                    class_checkpoint_path,
                    class_checkpoint,
                    contour_checkpoint_path,
                    contour_checkpoint,
                    output,
                )

    def test_transfer_sidecar_binds_checkpoint_index_fold_and_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "index.csv"
            checkpoint_path = root / "checkpoint.pt"
            prediction = root / "prediction.npz"
            sidecar = root / "transfer.json"
            index.write_text("patch_id,case\np,c\n", encoding="utf-8")
            checkpoint_path.write_bytes(b"checkpoint")
            prediction.write_bytes(b"prediction")
            index_digest = hashlib.sha256(index.read_bytes()).hexdigest()
            checkpoint = {
                "fold": 1,
                "folds": 5,
                "config": {"seed": 17},
                "index_sha256": "0" * 64,
            }
            payload = {
                "cross_index_transfer": True,
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_sha256": file_sha256(checkpoint_path),
                "training_index_sha256": checkpoint["index_sha256"],
                "evaluation_index": str(index.resolve()),
                "evaluation_index_sha256": index_digest,
                "predictions": str(prediction.resolve()),
                "predictions_sha256": file_sha256(prediction),
                "fold": 1,
                "folds": 5,
                "seed": 17,
                "metrics": {"selection_score": 0.5},
                "public_read": False,
                "hidden_read": False,
            }
            sidecar.write_text(json.dumps(payload), encoding="utf-8")
            validated = validate_transfer_result(
                sidecar, index, index_digest, checkpoint_path, checkpoint, prediction
            )
            self.assertEqual(validated["metrics"]["selection_score"], 0.5)
            payload["predictions_sha256"] = "f" * 64
            sidecar.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prediction_sha256"):
                validate_transfer_result(
                    sidecar, index, index_digest, checkpoint_path, checkpoint, prediction
                )

    def test_deployment_thresholds_remain_exact_near_upper_bound(self):
        self.assertEqual(deployment_classification_thresholds(0.95), [0.85, 0.9, 0.95])

    def test_classification_threshold_gives_each_case_equal_total_weight(self):
        expected = {}
        patch_ids = []
        labels = []
        probabilities = []
        for index in range(20):
            patch_id = f"large__c{index:04d}"
            expected[patch_id] = {"case": "large"}
            patch_ids.append(patch_id)
            labels.append(0)
            probabilities.append(0.60)
        expected["large__positive"] = {"case": "large"}
        patch_ids.append("large__positive")
        labels.append(1)
        probabilities.append(0.90)
        expected["small__positive"] = {"case": "small"}
        patch_ids.append("small__positive")
        labels.append(1)
        probabilities.append(0.55)
        expected["small__negative"] = {"case": "small"}
        patch_ids.append("small__negative")
        labels.append(0)
        probabilities.append(0.50)

        result = case_balanced_threshold_metrics(
            expected,
            patch_ids,
            np.asarray(labels, dtype=np.uint8),
            np.asarray(probabilities, dtype=np.float64),
        )

        self.assertAlmostEqual(result["threshold"], 0.55)
        self.assertEqual(result["cases"], 2)

    def test_segmentation_grid_expands_upward_from_observed_ceiling(self):
        self.assertEqual(deployment_segmentation_thresholds(0.80), [0.8, 0.85, 0.9])
        self.assertEqual(deployment_segmentation_thresholds(0.95), [0.85, 0.9, 0.95])
        self.assertEqual(deployment_segmentation_thresholds(0.60), [0.55, 0.6, 0.65])

    def test_exact_fold_coverage_and_global_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "index.csv"
            rows = [
                {"patch_id": "case0__c0001", "case": "case0", "cohort": "a", "label": 1},
                {"patch_id": "case0__c0002", "case": "case0", "cohort": "a", "label": 0},
                {"patch_id": "case1__c0001", "case": "case1", "cohort": "b", "label": 1},
                {"patch_id": "case1__c0002", "case": "case1", "cohort": "b", "label": 0},
            ]
            with index.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            digest = hashlib.sha256(index.read_bytes()).hexdigest()
            thresholds = np.linspace(0.20, 0.80, 13).astype(np.float32)
            checkpoints, predictions = [], []
            for fold in (0, 1):
                checkpoint = root / f"fold{fold}.pt"
                torch.save(
                    {
                        "fold": fold,
                        "epoch": 3,
                        "index_sha256": digest,
                        "metrics": {"selection_score": 0.8},
                    },
                    checkpoint,
                )
                prediction = root / f"fold{fold}.npz"
                grid = np.full((1, 13), 0.5, dtype=np.float32)
                grid[:, 6] = 0.9
                np.savez_compressed(
                    prediction,
                    patch_id=np.asarray([f"case{fold}__c0001", f"case{fold}__c0002"]),
                    label=np.asarray([1, 0], dtype=np.uint8),
                    classification_probability=np.asarray([0.9 - 0.1 * fold, 0.1 + 0.1 * fold]),
                    positive_patch_id=np.asarray([f"case{fold}__c0001"]),
                    segmentation_thresholds=thresholds,
                    segmentation_dice_grid=grid,
                )
                checkpoints.append(checkpoint)
                predictions.append(prediction)
            output = root / "report.json"
            arguments = ["summarize_candidate_refiner_oof.py", "--index", str(index)]
            for checkpoint, prediction in zip(checkpoints, predictions):
                arguments.extend(("--checkpoint", str(checkpoint), "--prediction", str(prediction)))
            arguments.extend(("--output", str(output), "--bootstrap", "20"))
            with patch.object(sys, "argv", arguments):
                main()
            report = json.loads(output.read_text())
            self.assertEqual(report["schema_version"], 1)
            implementation = Path(report["implementation"])
            self.assertEqual(
                report["implementation_sha256"], file_sha256(implementation)
            )
            for fold, checkpoint, prediction in zip(
                report["folds"], checkpoints, predictions
            ):
                self.assertEqual(fold["checkpoint_sha256"], file_sha256(checkpoint))
                self.assertEqual(fold["prediction_sha256"], file_sha256(prediction))
            self.assertTrue(report["coverage"]["every_patch_exactly_once"])
            self.assertEqual(report["coverage"]["patches"], 4)
            self.assertEqual(report["classification"]["average_precision"], 1.0)
            self.assertEqual(len(report["classification"]["deployment_thresholds"]), 3)
            self.assertAlmostEqual(report["segmentation"]["threshold"], 0.5)
            self.assertAlmostEqual(report["segmentation"]["dice"], 0.9, places=6)
            self.assertEqual(
                report["segmentation"]["aggregation"],
                "patches averaged within case, then cases averaged",
            )
            self.assertEqual(
                [row["cohort"] for row in report["cohort_summary"]], ["a", "b"]
            )
            self.assertEqual(
                sum(row["patches"] for row in report["cohort_summary"]), 4
            )

    def test_cohort_summary_uses_global_thresholds_and_exact_coverage(self):
        expected = {
            "a__c0": {"cohort": "A", "case": "a"},
            "a__c1": {"cohort": "A", "case": "a"},
            "b__c0": {"cohort": "B", "case": "b"},
            "b__c1": {"cohort": "B", "case": "b"},
        }
        result = cohort_metrics(
            expected,
            ["a__c0", "a__c1", "b__c0", "b__c1"],
            np.asarray([1, 0, 1, 0], dtype=np.uint8),
            np.asarray([0.9, 0.1, 0.4, 0.3]),
            0.5,
            ["a__c0", "b__c0"],
            np.asarray([[0.8, 0.7], [0.2, 0.1]]),
            0,
        )
        self.assertEqual([row["cohort"] for row in result], ["A", "B"])
        self.assertEqual(result[0]["classification_recall_at_global_threshold"], 1.0)
        self.assertEqual(result[1]["classification_recall_at_global_threshold"], 0.0)
        self.assertEqual(result[0]["segmentation_dice_at_global_threshold"], 0.8)
        self.assertEqual(result[1]["segmentation_dice_at_global_threshold"], 0.2)

    def test_source_specific_thresholds_use_exact_held_out_coverage(self):
        expected = {
            "a__c0": {"source_name": "epoch105", "case": "a"},
            "a__c1": {"source_name": "epoch105", "case": "a"},
            "b__c0": {"source_name": "weak", "case": "b"},
            "b__c1": {"source_name": "weak", "case": "b"},
        }
        result = source_classification_metrics(
            expected,
            list(expected),
            np.asarray([1, 0, 1, 0], dtype=np.uint8),
            np.asarray([0.9, 0.1, 0.7, 0.6]),
        )
        self.assertEqual(set(result), {"epoch105", "weak"})
        self.assertEqual(sum(row["patches"] for row in result.values()), 4)
        self.assertGreater(result["epoch105"]["threshold"], 0)
        self.assertGreater(result["weak"]["threshold"], 0)

    def test_source_specific_thresholds_reject_partial_identity(self):
        expected = {
            "a__c0": {"source_name": "epoch105", "case": "a"},
            "a__c1": {},
        }
        with self.assertRaisesRegex(ValueError, "source_name must cover every patch"):
            source_classification_metrics(
                expected,
                list(expected),
                np.asarray([1, 0], dtype=np.uint8),
                np.asarray([0.9, 0.1]),
            )

    def test_exchangeable_sources_share_one_pooled_threshold(self):
        expected = {
            "a0": {"source_name": "lung_cell0", "case": "a"},
            "a1": {"source_name": "lung_cell0", "case": "a"},
            "b0": {"source_name": "lung_cell1", "case": "b"},
            "b1": {"source_name": "lung_cell1", "case": "b"},
        }
        patch_ids = list(expected)
        labels = np.asarray([1, 0, 1, 0], dtype=np.uint8)
        probabilities = np.asarray([0.9, 0.8, 0.7, 0.1])
        reports = source_classification_metrics(
            expected, patch_ids, labels, probabilities
        )
        individual = {name: row["threshold"] for name, row in reports.items()}
        groups = pool_source_classification_groups(
            reports,
            expected,
            patch_ids,
            labels,
            probabilities,
            ["lung_cells=lung_cell0,lung_cell1"],
        )
        self.assertEqual(set(groups), {"lung_cells"})
        self.assertEqual(
            reports["lung_cell0"]["threshold"],
            reports["lung_cell1"]["threshold"],
        )
        self.assertEqual(reports["lung_cell0"]["threshold_group"], "lung_cells")
        self.assertEqual(reports["lung_cell0"]["individual_threshold"],
                         individual["lung_cell0"])

    def test_source_thresholds_equalize_case_weight(self):
        expected = {}
        patch_ids = []
        labels = []
        probabilities = []
        for index in range(20):
            patch_id = f"large__c{index:04d}"
            expected[patch_id] = {"source_name": "epoch105", "case": "large"}
            patch_ids.append(patch_id)
            labels.append(0)
            probabilities.append(0.60)
        for patch_id, label, probability in (
            ("large__positive", 1, 0.90),
            ("small__positive", 1, 0.55),
            ("small__negative", 0, 0.50),
        ):
            case = patch_id.split("__", 1)[0]
            expected[patch_id] = {"source_name": "epoch105", "case": case}
            patch_ids.append(patch_id)
            labels.append(label)
            probabilities.append(probability)
        result = source_classification_metrics(
            expected,
            patch_ids,
            np.asarray(labels, dtype=np.uint8),
            np.asarray(probabilities, dtype=np.float64),
        )
        self.assertAlmostEqual(result["epoch105"]["threshold"], 0.55)
        self.assertEqual(result["epoch105"]["cases"], 2)
        self.assertEqual(
            result["epoch105"]["aggregation"],
            "patches weighted to equal total weight per case",
        )

    def test_source_contours_are_case_balanced_and_exchangeable_cells_are_pooled(self):
        expected = {
            "a0": {"patch_id": "a0", "source_name": "lung_cell0", "case": "a", "cohort": "A", "label": "1"},
            "a1": {"patch_id": "a1", "source_name": "lung_cell0", "case": "a", "cohort": "A", "label": "1"},
            "b0": {"patch_id": "b0", "source_name": "lung_cell1", "case": "b", "cohort": "B", "label": "1"},
        }
        positive_ids = list(expected)
        thresholds = np.asarray([0.4, 0.5, 0.6])
        grid = np.asarray(
            [[0.9, 0.5, 0.2], [0.9, 0.5, 0.2], [0.1, 0.6, 0.95]]
        )
        reports = source_segmentation_metrics(
            expected, positive_ids, grid, thresholds
        )
        self.assertEqual(reports["lung_cell0"]["threshold"], 0.4)
        self.assertEqual(reports["lung_cell1"]["threshold"], 0.6)
        groups = pool_source_segmentation_groups(
            reports,
            expected,
            positive_ids,
            grid,
            thresholds,
            ["lung_cells=lung_cell0,lung_cell1"],
        )
        self.assertEqual(groups["lung_cells"]["positive_cases"], 2)
        self.assertEqual(
            reports["lung_cell0"]["threshold"], reports["lung_cell1"]["threshold"]
        )
        self.assertIn("individual_threshold", reports["lung_cell0"])


if __name__ == "__main__":
    unittest.main()
