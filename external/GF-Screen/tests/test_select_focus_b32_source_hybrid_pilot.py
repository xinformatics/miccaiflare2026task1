import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from focus_validation_metrics import aggregate_positive_dice
from select_focus_b32_source_hybrid_pilot import (
    EXPECTED_GATE,
    EXPECTED_SOURCES,
    EXPECTED_TRAINING,
    select,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FocusB32SourceHybridPilotTests(unittest.TestCase):
    def build(self, root: Path, weak_gain: float, lung_gain: float, weak_negative: int = 0):
        index = root / "index.csv"
        fields = ["patch_id", "case", "cohort", "label", "source_name"]
        index_rows = [
            {
                "patch_id": f"unused-{case}",
                "case": f"case-{case:04d}",
                "cohort": "all",
                "label": str(case % 2),
                "source_name": "epoch105",
            }
            for case in range(707)
        ]
        with index.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(index_rows)

        held_out = []
        sources = [source for group in EXPECTED_SOURCES.values() for source in sorted(group)]
        for position, source in enumerate(sources):
            for label in (1, 0):
                held_out.append(
                    {
                        "patch_id": f"p-{source}-{label}",
                        "case": f"held-{position:02d}",
                        "cohort": f"cohort-{position % 3}",
                        "label": str(label),
                        "source_name": source,
                    }
                )

        positive = [row["patch_id"] for row in held_out if int(row["label"]) == 1]
        negative = [row["patch_id"] for row in held_out if int(row["label"]) == 0]
        thresholds = np.asarray([0.5, 0.6], dtype=np.float32)
        b32_grid = np.full((len(positive), 2), 0.40, dtype=np.float32)
        focus_grid = np.empty_like(b32_grid)
        for index_position, patch_id in enumerate(positive):
            source = patch_id[len("p-") : -len("-1")]
            if source in EXPECTED_SOURCES["weak_sources"]:
                gain = weak_gain
            elif source in EXPECTED_SOURCES["lung_cells"]:
                gain = lung_gain
            else:
                gain = -0.03
            focus_grid[index_position] = (0.40 + gain, 0.39 + gain)
        negative_grid = np.zeros((len(negative), 2), dtype=np.uint8)
        if weak_negative:
            for position, patch_id in enumerate(negative):
                source = patch_id[len("p-") : -len("-0")]
                if source in EXPECTED_SOURCES["weak_sources"]:
                    negative_grid[position] = weak_negative

        baseline_prediction = root / "b32.npz"
        focus_prediction = root / "focus.npz"
        np.savez_compressed(
            baseline_prediction,
            positive_patch_id=np.asarray(positive),
            segmentation_thresholds=thresholds,
            segmentation_dice_grid=b32_grid,
        )
        np.savez_compressed(
            focus_prediction,
            positive_patch_id=np.asarray(positive),
            segmentation_thresholds=thresholds,
            segmentation_dice_grid=focus_grid,
            negative_patch_id=np.asarray(negative),
            negative_nonempty_grid=negative_grid,
        )
        focus_global = aggregate_positive_dice(
            positive,
            focus_grid,
            thresholds,
            held_out,
            negative_grid.sum(axis=0),
            len(negative),
        )
        b32_global = aggregate_positive_dice(positive, b32_grid, thresholds, held_out)
        baseline_checkpoint = root / "b32.pt"
        focus_checkpoint = root / "focus.pt"
        torch.save(
            {
                "fold": 0,
                "folds": 5,
                "index_sha256": sha(index),
                "config": {"base_channels": 32},
                "metrics": {
                    "segmentation_threshold": b32_global["threshold"],
                    "segmentation_dice": b32_global["case_balanced_dice"],
                    "cohort_balanced_segmentation_dice": b32_global[
                        "cohort_balanced_dice"
                    ],
                },
                "best_segmentation": b32_global["case_balanced_dice"],
            },
            baseline_checkpoint,
        )
        torch.save(
            {
                "fold": 0,
                "folds": 5,
                "index_sha256": sha(index),
                "config": dict(EXPECTED_TRAINING),
                "architecture": "FocusNet_soft_lung_seed",
                "pretrained_sha256": "released",
                "metrics": {
                    "segmentation_threshold": focus_global["threshold"],
                    "case_balanced_segmentation_dice": focus_global[
                        "case_balanced_dice"
                    ],
                    "cohort_balanced_segmentation_dice": focus_global[
                        "cohort_balanced_dice"
                    ],
                    "negative_nonempty_rate": focus_global[
                        "negative_nonempty_rate"
                    ],
                },
            },
            focus_checkpoint,
        )
        plan = root / "plan.json"
        plan.write_text(
            json.dumps(
                {
                    "index": str(index),
                    "index_sha256": sha(index),
                    "pretrained_sha256": "released",
                    "baseline_checkpoint_sha256": sha(baseline_checkpoint),
                    "baseline_prediction_sha256": sha(baseline_prediction),
                    "training": EXPECTED_TRAINING,
                    "mixed_source_gate": EXPECTED_GATE,
                }
            ),
            encoding="utf-8",
        )
        return (
            plan,
            focus_checkpoint,
            focus_prediction,
            baseline_checkpoint,
            baseline_prediction,
            held_out,
        )

    def run_case(self, weak_gain: float, lung_gain: float, weak_negative: int = 0):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        assets = self.build(Path(temporary.name), weak_gain, lung_gain, weak_negative)
        with patch(
            "select_focus_b32_source_hybrid_pilot.grouped_fold_rows",
            return_value=([], assets[-1]),
        ):
            return select(*assets[:-1])

    def test_source_gain_promotes_despite_global_mature_regression(self):
        result = self.run_case(weak_gain=0.04, lung_gain=0.03)
        self.assertEqual(result["decision"], "PROMOTE_MIXED_TO_FIVEFOLD")
        self.assertEqual(
            result["eligible_focus_groups"], ["lung_cells", "weak_sources"]
        )
        self.assertFalse(result["groups"]["mature"]["qualifies"])

    def test_single_metric_or_small_gain_does_not_promote(self):
        result = self.run_case(weak_gain=0.019, lung_gain=0.0)
        self.assertEqual(result["decision"], "KEEP_B32_MIXED")

    def test_source_specific_negative_activation_fails_closed(self):
        result = self.run_case(weak_gain=0.04, lung_gain=0.0, weak_negative=1)
        self.assertEqual(result["decision"], "KEEP_B32_MIXED")
        self.assertFalse(
            result["groups"]["weak_sources"]["checks"][
                "focus_negative_nonempty_rate"
            ]
        )


if __name__ == "__main__":
    unittest.main()
