import tempfile
import unittest
from pathlib import Path

import numpy as np

from select_candidate_refiner_small_lesion import (
    decide,
    source_dice_at_deployment_threshold,
)


def summary(segmentation, classification, composite):
    return {
        "segmentation": {"segmentation_dice": segmentation},
        "classification": {"average_precision": classification},
        "composite": {"selection_score": composite},
    }


class SmallLesionSelectionTests(unittest.TestCase):
    def test_source_dice_is_case_balanced_and_collapses_lung_cells(self):
        rows = [
            {"patch_id": "a", "case": "case1", "label": "1", "source_name": "lndb_weak"},
            {"patch_id": "b", "case": "case1", "label": "1", "source_name": "lndb_weak"},
            {"patch_id": "c", "case": "case2", "label": "1", "source_name": "lung_cell7"},
            {"patch_id": "n", "case": "case2", "label": "0", "source_name": "lung_cell0"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.npz"
            np.savez_compressed(
                path,
                positive_patch_id=np.asarray(["a", "b", "c"]),
                segmentation_thresholds=np.asarray([0.5, 0.6]),
                segmentation_dice_grid=np.asarray([[0.2, 0.3], [0.6, 0.7], [0.8, 0.9]]),
            )
            result = source_dice_at_deployment_threshold(path, rows, 0.6)
        self.assertAlmostEqual(result["lndb_weak"]["case_balanced_dice"], 0.5)
        self.assertAlmostEqual(result["lung_cells"]["case_balanced_dice"], 0.9)
        self.assertAlmostEqual(result["pooled_small_sources"]["case_balanced_dice"], 0.7)

    def test_all_targeted_and_global_gates_are_required(self):
        baseline = summary(0.50, 0.80, 0.65)
        candidate = summary(0.51, 0.795, 0.655)
        baseline_sources = {
            "pooled_small_sources": {"case_balanced_dice": 0.40},
            "lndb_weak": {"case_balanced_dice": 0.30},
            "lung_cells": {"case_balanced_dice": 0.50},
        }
        candidate_sources = {
            "pooled_small_sources": {"case_balanced_dice": 0.42},
            "lndb_weak": {"case_balanced_dice": 0.33},
            "lung_cells": {"case_balanced_dice": 0.51},
        }
        decision, _, gates = decide(
            baseline, candidate, baseline_sources, candidate_sources
        )
        self.assertEqual(decision, "EXPAND_SMALL_LESION_LOSS")
        self.assertTrue(all(gates.values()))
        candidate_sources["pooled_small_sources"]["case_balanced_dice"] = 0.405
        decision, _, gates = decide(
            baseline, candidate, baseline_sources, candidate_sources
        )
        self.assertEqual(decision, "KEEP_LEGACY_LOSS")
        self.assertFalse(gates["small_source_gain"])


if __name__ == "__main__":
    unittest.main()
