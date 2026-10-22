import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from select_focus_encoder_upgrade_fivefold import main


def focus(index_sha, case, cohort, negative, freeze, scale, weak_gain=0.0):
    groups = {
        "mature": {
            "dice": 0.55,
            "cohort_balanced_dice": 0.55,
            "selection_score": 0.53,
            "negative_nonempty_rate": 0.90,
        },
        "weak_sources": {
            "dice": 0.25 + weak_gain,
            "cohort_balanced_dice": 0.25 + weak_gain,
            "selection_score": 0.23 + weak_gain,
            "negative_nonempty_rate": 0.60,
        },
        "lung_cells": {
            "dice": 0.32,
            "cohort_balanced_dice": 0.32,
            "selection_score": 0.30,
            "negative_nonempty_rate": 0.82,
        },
    }
    return {
        "folds": 5,
        "seed": 20260823,
        "expected_epochs": 11,
        "index_sha256": index_sha,
        "training_recipe": {
            "epochs": 10,
            "batch_size": 8,
            "lr": 3e-5,
            "weight_decay": 1e-5,
            "loss_recipe": "release_dice_ce",
            "sampling_temperature": 0.5,
            "samples_per_epoch": 2048,
            "freeze_encoder": freeze,
            "encoder_lr_scale": scale,
            "input_mode": "soft_lung_seed",
            "segmentation_threshold_max": 0.95,
        },
        "segmentation": {
            "dice": case,
            "cohort_balanced_dice": cohort,
            "negative_nonempty_rate": negative,
        },
        "source_segmentation_groups": groups,
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
    }


class SelectFocusEncoderUpgradeFivefoldTests(unittest.TestCase):
    def run_selector(self, root, candidate, frozen):
        index_sha = candidate["index_sha256"]
        candidate_path = root / "candidate.json"
        frozen_path = root / "frozen.json"
        b32_path = root / "b32.json"
        output = root / "decision.json"
        b32 = {
            "index_sha256": index_sha,
            "coverage": {"every_patch_exactly_once": True, "cases": 707},
            "segmentation": {"dice": 0.398, "cohort_balanced_dice": 0.398},
        }
        for path, payload in (
            (candidate_path, candidate),
            (frozen_path, frozen),
            (b32_path, b32),
        ):
            path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        argv = [
            "selector",
            "--candidate-summary",
            str(candidate_path),
            "--frozen-summary",
            str(frozen_path),
            "--b32-summary",
            str(b32_path),
            "--output",
            str(output),
        ]
        with patch.object(sys, "argv", argv):
            main()
        return json.loads(output.read_text(encoding="utf-8"))

    def test_deployed_route_gain_promotes_even_if_global_focus_is_lower(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = focus("index", 0.423, 0.424, 0.83, True, 1.0)
            candidate = focus("index", 0.410, 0.411, 0.84, False, 0.1, 0.004)
            payload = self.run_selector(root, candidate, frozen)
            self.assertEqual(
                payload["decision"],
                "PROMOTE_ENCODER_UPGRADE_TO_ROUTED_FULL_VOLUME_OOF",
            )
            self.assertEqual(payload["qualified_routes"], ["weak_sources"])
            self.assertEqual(payload["mature_route"], "B32_UNCHANGED")
            self.assertLess(
                payload["global_focus_diagnostic_not_a_gate"][
                    "case_gain_over_frozen"
                ],
                0.0,
            )

    def test_rejects_when_no_deployed_route_has_paired_gain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = focus("index", 0.423, 0.424, 0.83, True, 1.0)
            candidate = focus("index", 0.440, 0.441, 0.82, False, 0.1, 0.001)
            payload = self.run_selector(root, candidate, frozen)
            self.assertEqual(payload["decision"], "KEEP_FROZEN_FOCUS")
            self.assertEqual(payload["qualified_routes"], [])


if __name__ == "__main__":
    unittest.main()
