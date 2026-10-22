import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from select_focus_encoder_upgrade_fold0_routes import select


class SelectFocusEncoderUpgradeFold0RouteTests(unittest.TestCase):
    def fixture(self, root: Path):
        index_sha = "index"
        training = {
            "fold": 0,
            "folds": 5,
            "epochs": 10,
            "batch_size": 8,
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
        plan = {
            "selection_basis": "test route gate",
            "index_sha256": index_sha,
            "training": training,
            "route_gate": {
                "minimum_case_gain": 0.005,
                "minimum_cohort_gain": 0.005,
                "minimum_selection_gain": 0.003,
                "maximum_negative_nonempty_increase": 0.02,
            },
        }
        checkpoint = root / "checkpoint.pt"
        torch.save(
            {
                "epoch": 4,
                "fold": 0,
                "folds": 5,
                "index_sha256": index_sha,
                "config": training,
            },
            checkpoint,
        )
        prediction = root / "prediction.npz"
        prediction.write_bytes(b"prediction")
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        prediction_sha = hashlib.sha256(prediction.read_bytes()).hexdigest()
        candidate_metrics = {
            "mature": {
                "selection_score": 0.40,
                "case_balanced_dice": 0.42,
                "cohort_balanced_dice": 0.42,
                "negative_nonempty_rate": 0.80,
            },
            "weak_sources": {
                "selection_score": 0.234,
                "case_balanced_dice": 0.247,
                "cohort_balanced_dice": 0.248,
                "negative_nonempty_rate": 0.69,
            },
            "lung_cells": {
                "selection_score": 0.290,
                "case_balanced_dice": 0.306,
                "cohort_balanced_dice": 0.309,
                "negative_nonempty_rate": 0.82,
            },
        }
        audit = root / "audit.json"
        audit.write_text(
            json.dumps(
                {
                    "epoch": 4,
                    "checkpoint_sha256": checkpoint_sha,
                    "prediction_sha256": prediction_sha,
                    "groups": candidate_metrics,
                }
            ),
            encoding="utf-8",
        )
        audit_sha = hashlib.sha256(audit.read_bytes()).hexdigest()
        selected = {}
        for route, metrics in candidate_metrics.items():
            selected[route] = {
                "epoch": 4,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha,
                "prediction": str(prediction),
                "prediction_sha256": prediction_sha,
                "audit": str(audit),
                "audit_sha256": audit_sha,
                "metrics": metrics,
            }
        candidate = {
            "fold": 0,
            "expected_epochs": 11,
            "seed": 20260823,
            "index_sha256": index_sha,
            "selected": selected,
            "public_read": False,
            "hidden_read": False,
            "submission_created": False,
        }
        frozen_metrics = {
            "mature": (0.40, 0.42, 0.42, 0.80),
            "weak_sources": (0.227, 0.241, 0.242, 0.68),
            "lung_cells": (0.296, 0.313, 0.316, 0.82),
        }
        frozen_selected = {}
        trajectory_groups = {}
        for route, values in frozen_metrics.items():
            selection_score, case_dice, cohort_dice, negative = values
            frozen_selected[route] = {
                "epoch": 5,
                "selection_score": selection_score,
                "case_balanced_dice": case_dice,
                "negative_nonempty_rate": negative,
            }
            trajectory_groups[route] = {
                "selection_score": selection_score,
                "case_balanced_dice": case_dice,
                "cohort_balanced_dice": cohort_dice,
                "negative_nonempty_rate": negative,
            }
        frozen = {
            "index_sha256": index_sha,
            "folds": 5,
            "seed": 20260823,
            "expected_epochs": 11,
            "training_recipe": {
                "freeze_encoder": True,
                "input_mode": "soft_lung_seed",
            },
            "fold_records": [
                {
                    "fold": 0,
                    "selected": frozen_selected,
                    "trajectory": [{"epoch": 5, "groups": trajectory_groups}],
                }
            ],
            "public_read": False,
            "hidden_read": False,
            "submission_created": False,
        }
        return plan, candidate, frozen

    def test_promotes_only_a_deployed_route_with_material_gain(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, candidate, frozen = self.fixture(Path(directory))
            result = select(plan, candidate, frozen)
            self.assertEqual(result["decision"], "PROMOTE_ROUTES_TO_FIVEFOLD")
            self.assertEqual(result["qualified_routes"], ["weak_sources"])
            self.assertEqual(result["mature_route"], "B32_UNCHANGED")

    def test_rejects_when_gain_is_only_mature(self):
        with tempfile.TemporaryDirectory() as directory:
            plan, candidate, frozen = self.fixture(Path(directory))
            candidate["selected"]["weak_sources"]["metrics"][
                "case_balanced_dice"
            ] = 0.242
            audit = Path(candidate["selected"]["weak_sources"]["audit"])
            audit_payload = json.loads(audit.read_text(encoding="utf-8"))
            audit_payload["groups"]["weak_sources"]["case_balanced_dice"] = 0.242
            audit.write_text(json.dumps(audit_payload), encoding="utf-8")
            digest = hashlib.sha256(audit.read_bytes()).hexdigest()
            for selected in candidate["selected"].values():
                selected["audit_sha256"] = digest
            result = select(plan, candidate, frozen)
            self.assertEqual(result["decision"], "KEEP_FROZEN_FOCUS")
            self.assertEqual(result["qualified_routes"], [])


if __name__ == "__main__":
    unittest.main()
