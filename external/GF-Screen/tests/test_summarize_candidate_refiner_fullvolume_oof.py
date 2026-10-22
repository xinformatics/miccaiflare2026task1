import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from summarize_candidate_refiner_fullvolume_oof import (
    bootstrap_selection_frequency,
    summarize,
)


class FullVolumeOOFSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        items = [
            {
                "name": f"case{index}",
                "cohort": "A" if index < 3 else "B",
                "complete_label": True,
            }
            for index in range(6)
        ]
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({"items": items}), encoding="utf-8")
        self.selection = self.root / "checkpoint.json"
        self.selection.write_text(
            json.dumps({"checkpoint_filename": "checkpoint_best_segmentation.pt"}),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def write_metrics(self, name, dsc, nsd, cases=None):
        path = self.root / f"{name}.csv"
        cases = cases or [f"case{index}" for index in range(6)]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("Name", "Lesion_DSC", "Lesion_NSD"))
            for case, x, y in zip(cases, dsc, nsd):
                writer.writerow((case, x, y))
        return path

    def write_summary(self, rows):
        path = self.root / "summary.tsv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(
                (
                    "variant",
                    "classification_threshold",
                    "segmentation_threshold",
                    "DSC",
                    "NSD",
                    "combined",
                    "metrics",
                )
            )
            writer.writerows(rows)
        return path

    def write_mode_summary(self, rows):
        path = self.root / "summary_modes.tsv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(
                (
                    "variant",
                    "classification_threshold",
                    "segmentation_threshold",
                    "deployment_mode",
                    "DSC",
                    "NSD",
                    "combined",
                    "metrics",
                )
            )
            writer.writerows(rows)
        return path

    def write_source_mode_summary(self, rows):
        path = self.root / "summary_source_modes.tsv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(
                (
                    "variant",
                    "classification_threshold",
                    "segmentation_threshold",
                    "deployment_mode",
                    "source_classification_thresholds",
                    "DSC",
                    "NSD",
                    "combined",
                    "metrics",
                )
            )
            writer.writerows(rows)
        return path

    def write_source_contour_summary(self, rows):
        path = self.root / "summary_source_contours.tsv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(
                (
                    "variant",
                    "classification_threshold",
                    "segmentation_threshold",
                    "deployment_mode",
                    "source_classification_thresholds",
                    "source_segmentation_thresholds",
                    "DSC",
                    "NSD",
                    "combined",
                    "metrics",
                )
            )
            writer.writerows(rows)
        return path

    def write_source_deployment_summary(self, rows):
        path = self.root / "summary_source_deployment.tsv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(
                (
                    "variant",
                    "classification_threshold",
                    "segmentation_threshold",
                    "deployment_mode",
                    "source_classification_thresholds",
                    "source_segmentation_thresholds",
                    "source_deployment_modes",
                    "DSC",
                    "NSD",
                    "combined",
                    "metrics",
                )
            )
            writer.writerows(rows)
        return path

    def write_geometry_summary(self, rows):
        path = self.root / "summary_geometry.tsv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(
                (
                    "variant",
                    "classification_threshold",
                    "segmentation_threshold",
                    "deployment_mode",
                    "source_classification_thresholds",
                    "source_segmentation_thresholds",
                    "source_deployment_modes",
                    "geometry_gates",
                    "DSC",
                    "NSD",
                    "combined",
                    "metrics",
                )
            )
            writer.writerows(rows)
        return path

    def test_selects_best_and_reports_stratified_uncertainty(self):
        first_dsc = [0.7, 0.8, 0.9, 0.6, 0.7, 0.8]
        first_nsd = [0.6, 0.7, 0.8, 0.5, 0.6, 0.7]
        second_dsc = [value - 0.05 for value in first_dsc]
        second_nsd = [value + 0.02 for value in first_nsd]
        third_dsc = [value - 0.10 for value in first_dsc]
        third_nsd = [value - 0.10 for value in first_nsd]
        first = self.write_metrics("first", first_dsc, first_nsd)
        second = self.write_metrics("second", second_dsc, second_nsd)
        third = self.write_metrics("third", third_dsc, third_nsd)
        summary = self.write_summary(
            [
                ("first", 0.7, 0.6, sum(first_dsc) / 6, sum(first_nsd) / 6, 0, first),
                ("second", 0.8, 0.6, sum(second_dsc) / 6, sum(second_nsd) / 6, 0, second),
                ("third", 0.9, 0.6, sum(third_dsc) / 6, sum(third_nsd) / 6, 0, third),
            ]
        )
        output = self.root / "result.json"
        payload = summarize(summary, self.selection, self.manifest, output, 500, 7)
        self.assertEqual(payload["variant"], "first")
        self.assertEqual(payload["cases"], 6)
        self.assertEqual(payload["cohorts"], 2)
        self.assertEqual(payload["bootstrap"]["method"], "case bootstrap stratified by source cohort")
        self.assertEqual(len(payload["selected"]["cohorts"]), 2)
        self.assertAlmostEqual(sum(payload["bootstrap_selection_frequency"].values()), 1.0)
        self.assertGreater(payload["paired_vs_runner_up"]["DSC_difference_95ci"]["low"], 0)
        self.assertEqual(json.loads(output.read_text())["classification_threshold"], 0.7)
        self.assertEqual(payload["deployment_mode"], "refined")
        self.assertEqual(payload["schema_version"], 1)
        for path_key, digest_key in (
            ("implementation", "implementation_sha256"),
            ("summary", "summary_sha256"),
            ("checkpoint_selection_path", "checkpoint_selection_sha256"),
            ("manifest", "manifest_sha256"),
        ):
            path = Path(payload[path_key])
            self.assertEqual(
                payload[digest_key], hashlib.sha256(path.read_bytes()).hexdigest()
            )

    def test_selects_from_complete_threshold_mode_grid(self):
        rows = []
        for threshold in (0.7, 0.8, 0.9):
            for mode in ("refined", "seed", "seed_union"):
                score = 0.55
                if threshold == 0.8 and mode == "seed_union":
                    score = 0.75
                metrics = self.write_metrics(
                    f"{threshold}_{mode}", [score] * 6, [score - 0.05] * 6
                )
                rows.append(
                    (
                        f"c{threshold}_{mode}", threshold, 0.6, mode,
                        score, score - 0.05, score - 0.025, metrics,
                    )
                )
        summary = self.write_mode_summary(rows)
        payload = summarize(
            summary, self.selection, self.manifest, self.root / "modes.json", 100, 9
        )
        self.assertEqual(payload["classification_threshold"], 0.8)
        self.assertEqual(payload["deployment_mode"], "seed_union")
        self.assertEqual(len(payload["all_variants"]), 9)
        self.assertIn("three prespecified output modes", payload["selection_basis"])

    def test_accepts_evidence_pruned_refined_only_grid(self):
        rows = []
        for threshold, score in ((0.7, 0.61), (0.8, 0.72), (0.9, 0.66)):
            metrics = self.write_metrics(
                f"refined_{threshold}", [score] * 6, [score - 0.05] * 6
            )
            rows.append(
                (
                    f"c{threshold}_refined", threshold, 0.6, "refined",
                    score, score - 0.05, score - 0.025, metrics,
                )
            )
        payload = summarize(
            self.write_mode_summary(rows), self.selection, self.manifest,
            self.root / "refined_only.json", 100, 13,
        )
        self.assertEqual(payload["classification_threshold"], 0.8)
        self.assertEqual(payload["deployment_mode"], "refined")
        self.assertEqual(len(payload["all_variants"]), 3)
        self.assertIn("fixed refined output mode", payload["selection_basis"])

    def test_accepts_refined_only_source_calibrated_grid(self):
        rows = []
        for threshold, score in ((0.7, 0.60), (0.8, 0.62), (0.9, 0.59)):
            metrics = self.write_metrics(
                f"global_refined_{threshold}", [score] * 6, [score - 0.05] * 6
            )
            rows.append(
                (
                    f"global_{threshold}_refined", threshold, 0.6, "refined", "{}",
                    score, score - 0.05, score - 0.025, metrics,
                )
            )
        mapping = json.dumps({"epoch105": 0.75, "weak": 0.9}, sort_keys=True)
        score = 0.74
        metrics = self.write_metrics(
            "source_refined", [score] * 6, [score - 0.05] * 6
        )
        rows.append(
            (
                "source_refined", 0.8, 0.6, "refined", mapping,
                score, score - 0.05, score - 0.025, metrics,
            )
        )
        payload = summarize(
            self.write_source_mode_summary(rows), self.selection, self.manifest,
            self.root / "source_refined.json", 100, 17,
        )
        self.assertEqual(payload["variant"], "source_refined")
        self.assertEqual(payload["source_classification_thresholds"],
                         {"epoch105": 0.75, "weak": 0.9})
        self.assertEqual(len(payload["all_variants"]), 4)
        self.assertIn("source-calibrated", payload["selection_basis"])

    def test_selects_patch_oof_declared_source_contours(self):
        rows = []
        for threshold, score in ((0.7, 0.60), (0.8, 0.62), (0.9, 0.59)):
            metrics = self.write_metrics(
                f"global_contour_{threshold}", [score] * 6, [score - 0.05] * 6
            )
            rows.append(
                (
                    f"global_{threshold}", threshold, 0.6, "refined", "{}", "{}",
                    score, score - 0.05, score - 0.025, metrics,
                )
            )
        class_mapping = {"epoch105": 0.75, "lung_cell0": 0.85}
        contour_mapping = {"epoch105": 0.65, "lung_cell0": 0.40}
        metrics = self.write_metrics("source_contours", [0.76] * 6, [0.71] * 6)
        rows.append(
            (
                "source_contours", 0.8, 0.6, "refined",
                json.dumps(class_mapping, sort_keys=True),
                json.dumps(contour_mapping, sort_keys=True),
                0.76, 0.71, 0.735, metrics,
            )
        )
        payload = summarize(
            self.write_source_contour_summary(rows), self.selection, self.manifest,
            self.root / "source_contours.json", 100, 31,
        )
        self.assertEqual(payload["variant"], "source_contours")
        self.assertEqual(payload["source_classification_thresholds"], class_mapping)
        self.assertEqual(payload["source_segmentation_thresholds"], contour_mapping)

    def test_selects_bounded_source_deployment_mode(self):
        rows = []
        for threshold, score in ((0.7, 0.60), (0.8, 0.62), (0.9, 0.59)):
            metrics = self.write_metrics(
                f"global_source_mode_{threshold}", [score] * 6, [score - 0.05] * 6
            )
            rows.append(
                (
                    f"global_{threshold}", threshold, 0.6, "refined", "{}", "{}", "{}",
                    score, score - 0.05, score - 0.025, metrics,
                )
            )
        modes = {"currentgf": "seed", "epoch105": "seed"}
        metrics = self.write_metrics("source_mode", [0.77] * 6, [0.72] * 6)
        rows.append(
            (
                "source_mode", 0.8, 0.6, "refined", "{}", "{}",
                json.dumps(modes, sort_keys=True), 0.77, 0.72, 0.745, metrics,
            )
        )
        payload = summarize(
            self.write_source_deployment_summary(rows), self.selection, self.manifest,
            self.root / "source_mode.json", 100, 37,
        )
        self.assertEqual(payload["variant"], "source_mode")
        self.assertEqual(payload["source_deployment_modes"], modes)

    def test_selects_and_serializes_predeclared_geometry_gates(self):
        rows = []
        for threshold, score in ((0.7, 0.60), (0.8, 0.62), (0.9, 0.59)):
            metrics = self.write_metrics(
                f"global_geometry_{threshold}", [score] * 6, [score - 0.05] * 6
            )
            rows.append(
                (
                    f"global_{threshold}", threshold, 0.6, "refined", "{}", "{}",
                    "{}", "{}", score, score - 0.05, score - 0.025, metrics,
                )
            )
        gates = {
            "min_seed_coverage": 0.1,
            "min_refined_overlap": 0.1,
            "min_volume_ratio": 0.25,
            "max_volume_ratio": 4.0,
        }
        metrics = self.write_metrics("geometry_mid", [0.77] * 6, [0.72] * 6)
        rows.append(
            (
                "geometry_mid", 0.8, 0.6, "refined", "{}", "{}", "{}",
                json.dumps(gates, sort_keys=True), 0.77, 0.72, 0.745, metrics,
            )
        )
        payload = summarize(
            self.write_geometry_summary(rows), self.selection, self.manifest,
            self.root / "geometry.json", 100, 41,
        )
        self.assertEqual(payload["variant"], "geometry_mid")
        self.assertEqual(payload["geometry_gates"], gates)
        self.assertIn("geometry-calibrated", payload["selection_basis"])

    def test_rejects_invalid_geometry_gate(self):
        metrics = self.write_metrics("invalid_geometry", [0.6] * 6, [0.55] * 6)
        rows = []
        for threshold in (0.7, 0.8, 0.9):
            rows.append(
                (
                    f"global_{threshold}", threshold, 0.6, "refined", "{}", "{}",
                    "{}", "{}", 0.6, 0.55, 0.575, metrics,
                )
            )
        rows.append(
            (
                "invalid", 0.8, 0.6, "refined", "{}", "{}", "{}",
                json.dumps({"min_seed_coverage": 1.1}),
                0.6, 0.55, 0.575, metrics,
            )
        )
        with self.assertRaisesRegex(ValueError, "overlap geometry"):
            summarize(
                self.write_geometry_summary(rows), self.selection, self.manifest,
                self.root / "invalid_geometry.json", 100, 43,
            )

    def test_accepts_second_predeclared_refined_source_recall_point(self):
        rows = []
        for threshold, score in ((0.7, 0.60), (0.8, 0.62), (0.9, 0.59)):
            metrics = self.write_metrics(
                f"global_two_source_{threshold}",
                [score] * 6,
                [score - 0.05] * 6,
            )
            rows.append(
                (
                    f"global_{threshold}_refined",
                    threshold,
                    0.6,
                    "refined",
                    "{}",
                    score,
                    score - 0.05,
                    score - 0.025,
                    metrics,
                )
            )
        for name, mapping, score in (
            ("source_precision", {"epoch105": 0.75, "lung_cell0": 0.85}, 0.70),
            ("source_recall", {"epoch105": 0.75, "lung_cell0": 0.75}, 0.76),
        ):
            metrics = self.write_metrics(
                name, [score] * 6, [score - 0.05] * 6
            )
            rows.append(
                (
                    name,
                    0.8,
                    0.6,
                    "refined",
                    json.dumps(mapping, sort_keys=True),
                    score,
                    score - 0.05,
                    score - 0.025,
                    metrics,
                )
            )
        payload = summarize(
            self.write_source_mode_summary(rows),
            self.selection,
            self.manifest,
            self.root / "two_source_refined.json",
            100,
            19,
        )
        self.assertEqual(payload["variant"], "source_recall")
        self.assertEqual(
            payload["source_classification_thresholds"],
            {"epoch105": 0.75, "lung_cell0": 0.75},
        )
        self.assertEqual(len(payload["all_variants"]), 5)

    def test_accepts_third_predeclared_refined_source_recall_point(self):
        rows = []
        for threshold, score in ((0.7, 0.60), (0.8, 0.62), (0.9, 0.59)):
            metrics = self.write_metrics(
                f"global_three_source_{threshold}",
                [score] * 6,
                [score - 0.05] * 6,
            )
            rows.append(
                (
                    f"global_{threshold}_refined", threshold, 0.6, "refined", "{}",
                    score, score - 0.05, score - 0.025, metrics,
                )
            )
        mappings = (
            ("source_precision", {"epoch105": 0.75, "lung_cell0": 0.85}, 0.70),
            ("source_recall", {"epoch105": 0.75, "lung_cell0": 0.75}, 0.74),
            ("source_high_recall", {"epoch105": 0.75, "lung_cell0": 0.65}, 0.78),
        )
        for name, mapping, score in mappings:
            metrics = self.write_metrics(name, [score] * 6, [score - 0.05] * 6)
            rows.append(
                (
                    name, 0.8, 0.6, "refined", json.dumps(mapping, sort_keys=True),
                    score, score - 0.05, score - 0.025, metrics,
                )
            )
        payload = summarize(
            self.write_source_mode_summary(rows),
            self.selection,
            self.manifest,
            self.root / "three_source_refined.json",
            100,
            23,
        )
        self.assertEqual(payload["variant"], "source_high_recall")
        self.assertEqual(
            payload["source_classification_thresholds"],
            {"epoch105": 0.75, "lung_cell0": 0.65},
        )
        self.assertEqual(len(payload["all_variants"]), 6)

    def test_rejects_tenth_source_operating_point(self):
        rows = []
        for threshold in (0.7, 0.8, 0.9):
            metrics = self.write_metrics(
                f"bounded_global_{threshold}", [0.6] * 6, [0.55] * 6
            )
            rows.append(
                (
                    f"global_{threshold}", threshold, 0.6, "refined", "{}",
                    0.6, 0.55, 0.575, metrics,
                )
            )
        for index, threshold in enumerate(
            (0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05)
        ):
            metrics = self.write_metrics(
                f"excess_source_{index}", [0.61] * 6, [0.56] * 6
            )
            rows.append(
                (
                    f"source_{index}", 0.8, 0.6, "refined",
                    json.dumps({"lung_cell0": threshold}),
                    0.61, 0.56, 0.585, metrics,
                )
            )
        with self.assertRaisesRegex(ValueError, "one to nine"):
            summarize(
                self.write_source_mode_summary(rows),
                self.selection,
                self.manifest,
                self.root / "too_many_source_points.json",
                100,
                29,
            )

    def test_selects_one_patch_oof_source_calibrated_operating_point(self):
        rows = []
        for threshold in (0.7, 0.8, 0.9):
            for mode in ("refined", "seed", "seed_union"):
                score = 0.60
                metrics = self.write_metrics(
                    f"global_{threshold}_{mode}", [score] * 6, [score - 0.05] * 6
                )
                rows.append(
                    (
                        f"global_{threshold}_{mode}", threshold, 0.6, mode, "{}",
                        score, score - 0.05, score - 0.025, metrics,
                    )
                )
        mapping = json.dumps({"epoch105": 0.75, "weak": 0.9}, sort_keys=True)
        for mode in ("refined", "seed", "seed_union"):
            score = 0.78 if mode == "seed" else 0.65
            metrics = self.write_metrics(
                f"source_{mode}", [score] * 6, [score - 0.05] * 6
            )
            rows.append(
                (
                    f"source_{mode}", 0.8, 0.6, mode, mapping,
                    score, score - 0.05, score - 0.025, metrics,
                )
            )
        payload = summarize(
            self.write_source_mode_summary(rows), self.selection, self.manifest,
            self.root / "source_modes.json", 100, 11,
        )
        self.assertEqual(payload["deployment_mode"], "seed")
        self.assertEqual(
            payload["source_classification_thresholds"],
            {"epoch105": 0.75, "weak": 0.9},
        )
        self.assertEqual(len(payload["all_variants"]), 12)
        self.assertIn("source-calibrated", payload["selection_basis"])

    def test_rejects_incomplete_case_coverage(self):
        metrics = self.write_metrics(
            "missing",
            [0.5] * 5,
            [0.5] * 5,
            cases=[f"case{index}" for index in range(5)],
        )
        rows = [
            (f"bad{index}", 0.5 + index * 0.1, 0.5, 0.5, 0.5, 0.5, metrics)
            for index in range(3)
        ]
        summary = self.write_summary(rows)
        with self.assertRaisesRegex(ValueError, "case coverage mismatch"):
            summarize(summary, self.selection, self.manifest, self.root / "out.json", 100, 1)

    def test_rejects_summary_metric_mismatch(self):
        metrics = self.write_metrics("mismatch", [0.5] * 6, [0.4] * 6)
        rows = [
            (f"bad{index}", 0.5 + index * 0.1, 0.5, 0.9, 0.4, 0.5, metrics)
            for index in range(3)
        ]
        summary = self.write_summary(rows)
        with self.assertRaisesRegex(ValueError, "summary mean mismatch"):
            summarize(summary, self.selection, self.manifest, self.root / "out.json", 100, 1)

    def test_fractionally_allocates_exact_bootstrap_ties(self):
        dsc = np.asarray([[0.7, 0.7, 0.6], [0.8, 0.7, 0.8]])
        nsd = np.asarray([[0.6, 0.6, 0.9], [0.5, 0.9, 0.5]])
        frequencies = bootstrap_selection_frequency(dsc, nsd)
        np.testing.assert_allclose(frequencies, [0.5, 0.25, 0.25])

    def test_requires_exactly_three_variants(self):
        metrics = self.write_metrics("only", [0.5] * 6, [0.4] * 6)
        summary = self.write_summary([("only", 0.5, 0.5, 0.5, 0.4, 0.45, metrics)])
        with self.assertRaisesRegex(ValueError, "exactly three"):
            summarize(summary, self.selection, self.manifest, self.root / "out.json", 100, 1)


if __name__ == "__main__":
    unittest.main()
