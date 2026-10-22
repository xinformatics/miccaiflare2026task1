import unittest
from pathlib import Path

from select_focus_prompt_pilot import select


class PromptFocusPilotSelectorTests(unittest.TestCase):
    def plan(self):
        return {
            "index_sha256": "index",
            "pretrained_sha256": "pretrained",
            "training": {
                "fold": 0,
                "folds": 5,
                "epochs": 10,
                "batch_size": 8,
                "workers": 2,
                "learning_rate": 3e-5,
                "weight_decay": 1e-5,
                "loss_recipe": "release_dice_ce",
                "sampling_temperature": 0.5,
                "samples_per_epoch": 2048,
                "seed": 20260823,
            },
            "fold0_gate": {
                "case_balanced_segmentation_dice_min": 0.55,
                "negative_nonempty_rate_max": 0.93,
                "selection_score_min": 0.531,
            },
        }

    def checkpoint(self):
        return {
            "epoch": 8,
            "fold": 0,
            "index_sha256": "index",
            "pretrained_sha256": "pretrained",
            "architecture": "GF Focus SwinUNETR contour refiner (soft_seed)",
            "config": {
                "fold": 0,
                "folds": 5,
                "epochs": 10,
                "batch_size": 8,
                "workers": 2,
                "lr": 3e-5,
                "weight_decay": 1e-5,
                "loss_recipe": "release_dice_ce",
                "sampling_temperature": 0.5,
                "samples_per_epoch": 2048,
                "freeze_encoder": True,
                "input_mode": "soft_seed",
                "seed": 20260823,
            },
            "metrics": {
                "case_balanced_segmentation_dice": 0.56,
                "segmentation_dice": 0.56,
                "negative_nonempty_rate": 0.90,
                "selection_score": 0.542,
            },
        }

    def test_all_gates_promote_to_confirmation_only(self):
        result = select(self.plan(), self.checkpoint(), Path("best.pt"))
        self.assertTrue(result["passed"])
        self.assertEqual(result["decision"], "CONFIRM_ON_UNTOUCHED_FOLD")
        self.assertFalse(result["submission_created"])

    def test_failed_negative_gate_keeps_base32(self):
        checkpoint = self.checkpoint()
        checkpoint["metrics"]["negative_nonempty_rate"] = 0.96
        result = select(self.plan(), checkpoint, Path("best.pt"))
        self.assertFalse(result["passed"])
        self.assertEqual(result["decision"], "KEEP_BASE32")

    def test_ct_only_checkpoint_fails_closed(self):
        checkpoint = self.checkpoint()
        checkpoint["config"]["input_mode"] = "soft"
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            select(self.plan(), checkpoint, Path("best.pt"))


if __name__ == "__main__":
    unittest.main()
