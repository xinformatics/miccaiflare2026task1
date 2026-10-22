from __future__ import annotations

import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from analyze_subtractive_component_gate_cv import (
    FEATURES,
    CaseStats,
    build_case_stats_from_rows,
    case_fold_map,
    crop_to_union,
    deletion_dsc_summary,
    dice,
    feature_matrix,
    load_anchor_metrics_cache,
    macro_dice_after_deletions,
    selected_deletions,
    score_official_case,
    stratified_bootstrap,
    stratified_bootstrap_one,
    table_dsc_metrics,
    threshold_grid,
    validate_row_identity,
)


def feature_row(case: str, component: int, is_tp: int = 0):
    row = {name: float(index + 1) for index, name in enumerate(FEATURES)}
    row.update({"case": case, "comp": component, "is_tp": is_tp})
    return row


class SubtractiveGateTests(unittest.TestCase):
    def test_audited_anchor_cache_skips_unchanged_volume_io(self) -> None:
        items = [{"name": "a", "cohort": "x"}]
        table = [{"case": "a", "anchor_DSC": 0.4}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "anchor.tsv"
            path.write_text(
                "case\tcohort\tanchor_DSC\tanchor_NSD\n"
                "a\tx\t0.4\t0.3\n",
                encoding="utf-8",
            )
            cache = load_anchor_metrics_cache(path, items, table)
            result = score_official_case(
                (items[0], str(Path(directory) / "missing"), [], 1000.0, cache["a"])
            )
        self.assertEqual(result["anchor_DSC"], 0.4)
        self.assertEqual(result["candidate_NSD"], 0.3)
        self.assertEqual(result["deleted_components"], 0)

    def test_union_crop_preserves_masks_and_component_ids(self) -> None:
        from scipy import ndimage

        target = np.zeros((20, 30, 12), dtype=bool)
        prediction = np.zeros_like(target)
        prediction[5:8, 9:12, 2:4] = True
        prediction[12:15, 20:23, 7:10] = True
        target[4:9, 8:13, 1:5] = True
        target[17:19, 25:28, 9:11] = True
        cropped_target, cropped_prediction = crop_to_union(target, prediction)
        self.assertEqual(int(cropped_target.sum()), int(target.sum()))
        self.assertEqual(int(cropped_prediction.sum()), int(prediction.sum()))
        structure = np.ones((3, 3, 3), dtype=np.uint8)
        full_labels, full_count = ndimage.label(prediction, structure=structure)
        crop_labels, crop_count = ndimage.label(cropped_prediction, structure=structure)
        self.assertEqual(full_count, crop_count)
        self.assertEqual(
            sorted(np.bincount(full_labels.ravel())[1:].tolist()),
            sorted(np.bincount(crop_labels.ravel())[1:].tolist()),
        )
        metric_root = Path(__file__).resolve().parents[3] / "handoff_bundle" / "code"
        if str(metric_root) not in sys.path:
            sys.path.insert(0, str(metric_root))
        from evaluate import score_case

        full_score = score_case(target.astype(np.uint8), prediction.astype(np.uint8), (1, 1, 2))
        crop_score = score_case(
            cropped_target.astype(np.uint8),
            cropped_prediction.astype(np.uint8),
            (1, 1, 2),
        )
        self.assertEqual(full_score, crop_score)
        empty_target, empty_prediction = crop_to_union(
            np.zeros((3, 4, 5), dtype=bool), np.zeros((3, 4, 5), dtype=bool)
        )
        self.assertEqual(empty_target.shape, (1, 1, 1))
        self.assertFalse(empty_prediction.any())

    def test_table_stats_reproduce_frozen_case_dice_without_volumes(self) -> None:
        # Prediction=100 mm3, target=80 mm3, overlap=60 mm3 -> Dice=2/3.
        rows = [
            {
                **feature_row("a", 1, 0),
                "mm3": 20.0,
                "overlap_frac": 0.0,
                "case_pred_mm3": 100.0,
                "base_dice": 2.0 / 3.0,
            },
            {
                **feature_row("a", 2, 1),
                "mm3": 80.0,
                "overlap_frac": 0.75,
                "case_pred_mm3": 100.0,
                "base_dice": 2.0 / 3.0,
            },
        ]
        items = [
            {"name": "a", "cohort": "x"},
            {"name": "empty-pred", "cohort": "x"},
        ]
        stats = build_case_stats_from_rows(items, rows)
        self.assertAlmostEqual(stats["a"].target_voxels, 80.0)
        self.assertAlmostEqual(dice(stats["a"], set()), 2.0 / 3.0)
        rows[0]["oof_delete"] = 1
        rows[1]["oof_delete"] = 0
        metrics = table_dsc_metrics(items, rows, stats)
        self.assertAlmostEqual(metrics[0]["candidate_DSC"], 120.0 / 160.0)
        self.assertEqual(metrics[1]["candidate_DSC"], 0.0)

    def test_case_fold_map_is_deterministic_balanced_and_grouped(self) -> None:
        items = [
            {"name": f"a{index}", "cohort": "a"} for index in range(11)
        ] + [{"name": f"b{index}", "cohort": "b"} for index in range(7)]
        first = case_fold_map(items, 5, 17)
        second = case_fold_map(list(reversed(items)), 5, 17)
        self.assertEqual(first, second)
        for cohort in ("a", "b"):
            counts = [
                sum(
                    first[str(item["name"])] == fold
                    for item in items
                    if item["cohort"] == cohort
                )
                for fold in range(5)
            ]
            self.assertLessEqual(max(counts) - min(counts), 1)

    def test_component_deletion_dice_and_conservative_threshold(self) -> None:
        stats = {
            "case": CaseStats(
                pred_voxels=100,
                target_voxels=80,
                overlap_voxels=60,
                component_voxels={1: 20, 2: 30},
                component_overlap={1: 0, 2: 25},
            )
        }
        rows = [feature_row("case", 1), feature_row("case", 2, is_tp=1)]
        probability = np.asarray([0.95, 0.20])
        self.assertTrue(math.isclose(dice(stats["case"], set()), 120 / 180))
        self.assertTrue(math.isclose(dice(stats["case"], {1}), 120 / 160))
        self.assertTrue(
            math.isclose(
                macro_dice_after_deletions(
                    rows, probability, 0.90, {"case"}, stats
                ),
                120 / 160,
            )
        )
        self.assertTrue(
            math.isclose(
                macro_dice_after_deletions(
                    rows, probability, 1.01, {"case"}, stats
                ),
                120 / 180,
            )
        )
        self.assertAlmostEqual(float(threshold_grid()[-1]), 1.01)
        score, catastrophic = deletion_dsc_summary(
            rows, probability, 0.90, {"case"}, stats
        )
        self.assertTrue(math.isclose(score, 120 / 160))
        self.assertEqual(catastrophic, 0.0)

    def test_selected_deletions_are_grouped_by_case(self) -> None:
        rows = [feature_row("a", 1), feature_row("a", 2), feature_row("b", 4)]
        rows[0]["mm3"] = 1500.0
        rows[1]["mm3"] = 2500.0
        rows[2]["mm3"] = 5000.0
        result = selected_deletions(rows, np.asarray([0.8, 0.2, 0.9]), 0.5)
        self.assertEqual(result, {"a": {1}, "b": {4}})
        traceable = selected_deletions(
            rows,
            np.asarray([0.8, 0.2, 0.9]),
            0.5,
            minimum_deletion_mm3_exclusive=2000.0,
        )
        self.assertEqual(traceable, {"b": {4}})

    def test_feature_matrix_uses_fixed_nonfinite_sentinel(self) -> None:
        row = feature_row("a", 1)
        row[FEATURES[0]] = float("nan")
        row[FEATURES[1]] = float("inf")
        row[FEATURES[2]] = float("-inf")
        matrix = feature_matrix([row])
        self.assertEqual(matrix.shape, (1, len(FEATURES)))
        self.assertEqual(matrix[0, :3].tolist(), [-2048.0, 4096.0, -4096.0])
        self.assertTrue(np.isfinite(matrix).all())

    def test_row_identity_requires_every_deployed_component_once(self) -> None:
        stats = {
            "a": CaseStats(3, 1, 1, {1: 2, 3: 1}, {1: 1, 3: 0}),
            "b": CaseStats(0, 0, 0, {}, {}),
        }
        rows = [feature_row("a", 1, 1), feature_row("a", 3, 0)]
        validate_row_identity(rows, stats)
        with self.assertRaisesRegex(ValueError, "does not reproduce"):
            validate_row_identity(rows[:1], stats)

    def test_stratified_bootstrap_reports_paired_gain_interval(self) -> None:
        metrics = []
        for cohort in ("a", "b"):
            for _ in range(5):
                metrics.append(
                    {
                        "cohort": cohort,
                        "anchor_DSC": 0.5,
                        "candidate_DSC": 0.6,
                        "anchor_NSD": 0.4,
                        "candidate_NSD": 0.45,
                    }
                )
        intervals = stratified_bootstrap(metrics, seed=3, iterations=1000)
        for value in intervals["DSC"]:
            self.assertAlmostEqual(value, 0.1)
        for value in intervals["NSD"]:
            self.assertAlmostEqual(value, 0.05)
        one = stratified_bootstrap_one(metrics, "DSC", seed=3, iterations=1000)
        for value in one:
            self.assertAlmostEqual(value, 0.1)


if __name__ == "__main__":
    unittest.main()
