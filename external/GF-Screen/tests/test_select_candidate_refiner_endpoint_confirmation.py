import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from select_candidate_refiner_endpoint_confirmation import (
    compare_candidate,
    selection_payload,
    trigger_payload,
)


CRITERIA = {
    "minimum_mean_DSC_gain": 0.002,
    "minimum_mean_NSD_gain": 0.0,
    "minimum_DSC_paired_bootstrap_95ci_lower": 0.0,
    "minimum_NSD_paired_bootstrap_95ci_lower": -0.001,
    "minimum_combined_case_win_rate": 0.52,
    "maximum_DSC_loss_gt_0.05_rate": 0.05,
    "minimum_per_cohort_DSC_delta": -0.005,
    "minimum_per_cohort_NSD_delta": -0.005,
}


class EndpointConfirmationTests(unittest.TestCase):
    def test_trigger_passes_only_at_frozen_anchor_endpoint(self):
        plan = {"trigger": {
            "anchor_currentgf_selected_segmentation_threshold": 0.95,
            "minimum_anchor_DSC_gain_0.90_to_0.95": 0.005,
            "minimum_anchor_NSD_gain_0.90_to_0.95": 0.0,
        }}
        sweep = {0.90: {"DSC": "0.50", "NSD": "0.40"},
                 0.95: {"DSC": "0.51", "NSD": "0.405"}}
        result = trigger_payload(plan, {"selected_segmentation_threshold": 0.95}, sweep)
        self.assertEqual(result["decision"], "RUN_ENDPOINT_CONFIRMATION")
        result = trigger_payload(plan, {"selected_segmentation_threshold": 0.90}, sweep)
        self.assertEqual(result["decision"], "SKIP_ENDPOINT_CONFIRMATION")

    def test_paired_gate_accepts_broad_safe_gain(self):
        anchor = np.full((40, 2), (0.4, 0.3))
        candidate = anchor + np.asarray((0.01, 0.005))
        cohorts = np.asarray(["a"] * 20 + ["b"] * 20)
        report = compare_candidate(anchor, candidate, cohorts, CRITERIA, 1000, 7)
        self.assertTrue(report["eligible"])

    def test_paired_gate_rejects_cohort_regression(self):
        anchor = np.full((40, 2), (0.4, 0.3))
        candidate = anchor.copy()
        candidate[:20] += np.asarray((-0.01, 0.0))
        candidate[20:] += np.asarray((0.04, 0.02))
        cohorts = np.asarray(["a"] * 20 + ["b"] * 20)
        report = compare_candidate(anchor, candidate, cohorts, CRITERIA, 1000, 7)
        self.assertFalse(report["eligible"])
        self.assertFalse(report["checks"]["per_cohort_dsc"])

    def test_paired_gate_rejects_large_dsc_loss_tail(self):
        anchor = np.full((40, 2), (0.4, 0.3))
        candidate = anchor + np.asarray((0.02, 0.01))
        candidate[:3, 0] = anchor[:3, 0] - 0.06
        cohorts = np.asarray(["a"] * 20 + ["b"] * 20)
        report = compare_candidate(anchor, candidate, cohorts, CRITERIA, 1000, 7)
        self.assertFalse(report["eligible"])
        self.assertFalse(report["checks"]["dsc_large_loss_rate"])

    def test_selection_prefers_best_eligible_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = [f"case{i}" for i in range(40)]
            paths = {}
            for threshold, gain in ((0.95, 0.0), (0.96, 0.006), (0.97, 0.012), (0.98, -0.01)):
                path = root / f"{threshold}.csv"
                with path.open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(("Name", *METRICS))
                    for name in names:
                        writer.writerow((name, 0.4 + gain, 0.3 + max(gain, 0) / 2))
                paths[threshold] = path
            manifest = {"cases": 40, "items": [
                {"name": name, "cohort": "a" if i < 20 else "b"}
                for i, name in enumerate(names)
            ]}
            plan = {"promotion_vs_0.95": CRITERIA}
            result = selection_payload(plan, manifest, paths, 1000, 11)
            self.assertEqual(result["decision"], "PROMOTE_ENDPOINT")
            self.assertEqual(result["selected_segmentation_threshold"], 0.97)


METRICS = ("Lesion_DSC", "Lesion_NSD")


if __name__ == "__main__":
    unittest.main()
