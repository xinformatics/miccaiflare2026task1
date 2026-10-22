import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from select_focus_discriminative_pilot import main


class SelectFocusDiscriminativePilotTests(unittest.TestCase):
    def test_requires_gain_over_b32_and_frozen_focus(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "index.csv"
            index.write_text("patch_id,case\np,c\n", encoding="utf-8")
            digest = hashlib.sha256(index.read_bytes()).hexdigest()
            frozen_decision = root / "frozen.json"
            frozen_decision.write_text(
                json.dumps({"decision": "KEEP_B32"}) + "\n", encoding="utf-8"
            )
            training = {
                "fold": 0,
                "folds": 5,
                "epochs": 10,
                "batch_size": 4,
                "workers": 2,
                "lr": 3e-5,
                "encoder_lr_scale": 0.1,
                "weight_decay": 1e-5,
                "loss_recipe": "release_dice_ce",
                "sampling_temperature": 0.5,
                "samples_per_epoch": 2048,
                "freeze_encoder": False,
                "input_mode": "soft_lung_seed",
                "segmentation_threshold_max": 0.95,
                "seed": 20260823,
            }
            baseline_path = root / "baseline.pt"
            frozen_path = root / "frozen.pt"
            candidate_path = root / "candidate.pt"
            baseline = {
                "fold": 0,
                "folds": 5,
                "index_sha256": digest,
                "config": {"base_channels": 32},
                "best_segmentation": 0.50,
                "metrics": {
                    "segmentation_dice": 0.50,
                    "cohort_balanced_segmentation_dice": 0.49,
                },
            }
            frozen = {
                "fold": 0,
                "folds": 5,
                "index_sha256": digest,
                "config": {
                    "freeze_encoder": True,
                    "input_mode": "soft_lung_seed",
                },
                "metrics": {
                    "case_balanced_segmentation_dice": 0.51,
                    "cohort_balanced_segmentation_dice": 0.50,
                },
            }
            candidate = {
                "fold": 0,
                "folds": 5,
                "index_sha256": digest,
                "pretrained_sha256": "pretrained",
                "config": dict(training),
                "architecture": "GF Focus SwinUNETR contour refiner (soft_lung_seed)",
                "history": [{"epoch": 0}],
                "metrics": {
                    "case_balanced_segmentation_dice": 0.52,
                    "cohort_balanced_segmentation_dice": 0.51,
                    "negative_nonempty_rate": 0.90,
                },
            }
            torch.save(baseline, baseline_path)
            torch.save(frozen, frozen_path)
            torch.save(candidate, candidate_path)
            file_sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            plan = root / "plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "training": training,
                        "index_sha256": digest,
                        "pretrained_sha256": "pretrained",
                        "baseline_checkpoint_sha256": file_sha(baseline_path),
                        "frozen_checkpoint_sha256": file_sha(frozen_path),
                        "frozen_decision": str(frozen_decision),
                        "frozen_decision_sha256": file_sha(frozen_decision),
                        "fold0_gate": {
                            "minimum_case_dice_gain_over_b32": 0.01,
                            "minimum_cohort_dice_gain_over_b32": 0.01,
                            "minimum_case_dice_gain_over_frozen": 0.005,
                            "minimum_cohort_dice_gain_over_frozen": 0.005,
                            "maximum_negative_nonempty_rate": 0.95,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "decision.json"
            argv = [
                "selector",
                "--plan",
                str(plan),
                "--candidate-checkpoint",
                str(candidate_path),
                "--baseline-checkpoint",
                str(baseline_path),
                "--frozen-checkpoint",
                str(frozen_path),
                "--output",
                str(output),
            ]
            with patch.object(sys, "argv", argv):
                main()
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["decision"], "PROMOTE_TO_FIVEFOLD")
            self.assertTrue(all(payload["checks"].values()))

            candidate["metrics"]["case_balanced_segmentation_dice"] = 0.512
            torch.save(candidate, candidate_path)
            with patch.object(sys, "argv", argv):
                main()
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["decision"], "KEEP_FROZEN_OR_B32")
            self.assertFalse(payload["checks"]["case_gain_over_frozen"])

            # A separately frozen batch-8 upgrade may use the same strict
            # selector without relaxing the historical batch-4 fallback.
            plan_payload = json.loads(plan.read_text(encoding="utf-8"))
            plan_payload["training"]["batch_size"] = 8
            plan.write_text(json.dumps(plan_payload) + "\n", encoding="utf-8")
            candidate["config"]["batch_size"] = 8
            candidate["metrics"]["case_balanced_segmentation_dice"] = 0.52
            torch.save(candidate, candidate_path)
            batch8_argv = (
                argv[:-2] + ["--expected-batch-size", "8"] + argv[-2:]
            )
            with patch.object(sys, "argv", batch8_argv):
                main()
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["decision"], "PROMOTE_TO_FIVEFOLD")
            self.assertTrue(all(payload["checks"].values()))


if __name__ == "__main__":
    unittest.main()
