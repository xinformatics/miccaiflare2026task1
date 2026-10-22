import unittest
from pathlib import Path

from select_candidate_refiner_base48_pilot import select


class Base48PilotSelectorTests(unittest.TestCase):
    def plan(self):
        return {
            "index_sha256": "abc",
            "candidate": {
                "base_channels": 48,
                "epochs": 12,
                "batch_size": 8,
                "learning_rate": 3e-4,
                "weight_decay": 1e-4,
                "sampling_temperature": 0.0,
                "seed": 20260822,
            },
            "promotion_gate": {
                "case_balanced_segmentation_dice_min": 0.53,
                "case_balanced_average_precision_min": 0.81,
                "case_balanced_selection_score_min": 0.67,
            },
        }

    def checkpoint(self):
        return {
            "epoch": 9,
            "fold": 0,
            "folds": 5,
            "index_sha256": "abc",
            "config": {
                "base_channels": 48,
                "epochs": 12,
                "batch_size": 8,
                "lr": 3e-4,
                "weight_decay": 1e-4,
                "sampling_temperature": 0.0,
                "seed": 20260822,
            },
            "metrics": {
                "aggregation": "patches weighted to equal total weight per case",
                "segmentation_dice": 0.54,
                "average_precision": 0.82,
                "selection_score": 0.68,
            },
        }

    def test_pass_requires_all_three_gates(self):
        result = select(self.plan(), self.checkpoint(), Path("best.pt"))
        self.assertTrue(result["passed"])
        self.assertEqual(result["decision"], "CONFIRM_ON_UNTOUCHED_FOLD")
        self.assertFalse(result["public_read"])

    def test_one_failed_gate_keeps_base32(self):
        checkpoint = self.checkpoint()
        checkpoint["metrics"]["average_precision"] = 0.80
        result = select(self.plan(), checkpoint, Path("best.pt"))
        self.assertFalse(result["passed"])
        self.assertEqual(result["decision"], "KEEP_BASE32")

    def test_configuration_mismatch_fails_closed(self):
        checkpoint = self.checkpoint()
        checkpoint["config"]["base_channels"] = 32
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            select(self.plan(), checkpoint, Path("best.pt"))


if __name__ == "__main__":
    unittest.main()
