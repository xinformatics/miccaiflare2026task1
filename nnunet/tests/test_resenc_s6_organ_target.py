import os
import sys
import unittest
import importlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("nnUNet_raw", str(PROJECT_ROOT / "nnUNet_raw"))
os.environ.setdefault("nnUNet_preprocessed", str(PROJECT_ROOT / "nnUNet_preprocessed"))
os.environ.setdefault("nnUNet_results", str(PROJECT_ROOT / "nnUNet_results"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-resenc-organ-tests")
sys.path.insert(0, str(PROJECT_ROOT))

from nnunet.resenc_s6_organ_target import (  # noqa: E402
    EXPECTED_SORTED_IDENTIFIER_SHA256,
    ORGAN_TARGET_EXPOSURE,
    nnUNetTrainerResEncS6PanCancerOrganTarget,
    organ_target_loader_factory,
    organ_target_sampling_probabilities,
)
from nnunet.run_resenc_s6_organ_target import resolve_trainer  # noqa: E402
from nnunet.select_resenc_s6_organ_public import load_policy  # noqa: E402
from nnunet.predict_resenc_s6_organ_safe import (  # noqa: E402
    TRAINER_NAME as PREDICTOR_TRAINER_NAME,
    terminal_argument,
)
from STUNetTrainerPanCancerSourceBalanced import identifier_inventory_sha256  # noqa: E402


class FakeDataset:
    def __init__(self, identifiers, foreground_counts=None):
        self.identifiers = tuple(identifiers)
        self.foreground_counts = foreground_counts


class FakeLoader:
    def __init__(self, data, *args, sampling_probabilities=None, **kwargs):
        del args, kwargs
        self._data = data
        self.sampling_probabilities = sampling_probabilities


class ResEncS6OrganTargetTests(unittest.TestCase):
    def test_runner_resolves_exact_workspace_trainer(self):
        self.assertIs(
            resolve_trainer(None, "nnUNetTrainerResEncS6PanCancerOrganTarget", None),
            nnUNetTrainerResEncS6PanCancerOrganTarget,
        )

    def test_fixed_one_percent_target_exposure(self):
        identifiers = (
            "Liver-A_001",
            "Kidney-B_001",
            "PseudoOrgan-AMOS22_0001",
            "PseudoOrgan-AMOS22_0002",
            "PseudoOrgan-AMOS22_0003",
            "PseudoOrgan-AMOS22_0004",
        )
        probabilities = organ_target_sampling_probabilities(
            identifiers, [50, 500, 100, 100, 100, 100]
        )
        target = np.asarray([item.startswith("PseudoOrgan-") for item in identifiers])
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertAlmostEqual(float(probabilities[target].sum()), ORGAN_TARGET_EXPOSURE)
        self.assertTrue(np.all(probabilities > 0))
        self.assertTrue(np.allclose(probabilities[target], ORGAN_TARGET_EXPOSURE / 4))

    def test_sampler_rejects_wrong_target_count_and_bad_alignment(self):
        with self.assertRaises(ValueError):
            organ_target_sampling_probabilities(
                ["Liver_001", "PseudoOrgan-AMOS22_0001"], [10, 10]
            )
        with self.assertRaises(ValueError):
            organ_target_sampling_probabilities(
                ["Liver_001"] + [f"PseudoOrgan-AMOS22_{i:04d}" for i in range(1, 5)],
                [10],
            )

    def test_loader_factory_changes_training_only(self):
        identifiers = ["Liver_001"] + [f"PseudoOrgan-AMOS22_{i:04d}" for i in range(1, 5)]
        resolver = lambda data: np.asarray(data.foreground_counts)
        factory = organ_target_loader_factory(FakeLoader, 0.25, resolver)
        training = factory(FakeDataset(identifiers, [100] * 5), sampling_probabilities=None)
        validation = factory(FakeDataset(["Liver_002"], [100]), sampling_probabilities=None)
        self.assertIsInstance(training, FakeLoader)
        self.assertAlmostEqual(
            float(training.sampling_probabilities[-4:].sum()), ORGAN_TARGET_EXPOSURE
        )
        self.assertEqual(training.lesion_uniform_center_fraction, 0.25)
        self.assertIsNone(validation.sampling_probabilities)

    def test_materialized_dataset_identity_is_frozen(self):
        raw = PROJECT_ROOT / "nnUNet_raw/Dataset503_PanCancerOrganTarget/imagesTr"
        identifiers = sorted(
            path.name.removesuffix("_0000.nii.gz") for path in raw.glob("*_0000.nii.gz")
        )
        self.assertEqual(len(identifiers), 17_579)
        self.assertEqual(
            identifier_inventory_sha256(identifiers), EXPECTED_SORTED_IDENTIFIER_SHA256
        )

    def test_trainer_recipe_is_short_and_cautious(self):
        preprocessed = PROJECT_ROOT / "nnUNet_preprocessed/Dataset503_PanCancerOrganTarget"
        plans = json.loads((preprocessed / "nnUNetPlans.json").read_text())
        dataset = json.loads((preprocessed / "dataset.json").read_text())
        trainer_module = importlib.import_module(
            "nnunetv2.training.nnUNetTrainer.nnUNetTrainer"
        )
        with (
            patch.object(
                trainer_module,
                "nnUNet_preprocessed",
                str(PROJECT_ROOT / "nnUNet_preprocessed"),
            ),
            patch.object(
                trainer_module,
                "nnUNet_results",
                str(PROJECT_ROOT / "nnUNet_results"),
            ),
        ):
            trainer = nnUNetTrainerResEncS6PanCancerOrganTarget(
                plans,
                "3d_fullres_ResEncS6",
                "all",
                dataset,
                device=torch.device("cpu"),
            )
        self.assertEqual(trainer.initial_lr, 3e-5)
        self.assertEqual(trainer.num_epochs, 25)
        self.assertEqual(trainer.organ_stage_end_epochs, (10, 25))
        self.assertEqual(trainer.lr_schedule_horizon_epochs, 25)
        self.assertEqual(trainer.configuration_manager.batch_size, 16)
        self.assertIn("Dataset503_PanCancerOrganTarget", trainer.output_folder)

    def test_public_policy_is_frozen_before_training(self):
        policy = load_policy(
            PROJECT_ROOT
            / "nnunet/RESENC_S6_ORGAN_TARGET_PUBLIC_SELECTION_POLICY_20260827.json"
        )
        self.assertEqual(policy["checkpoint_epochs"], [10, 25])
        self.assertEqual(
            [item["name"] for item in policy["deployments"]],
            [
                "resenc_standalone_p50",
                "gf_anchor_confirm_either_p70",
                "gf_resenc_boundary3mm_p70p50_ftu",
                "gf_union_resenc_p40_ftu_s10_mm3000",
            ],
        )
        self.assertEqual(
            policy["selection"]["qualification"]["minimum_fixed_public_DSC"],
            0.6366639257651584,
        )
        self.assertFalse(policy["threshold_search_after_public_read"])

    def test_preprocessing_is_serialized_after_positive_public_selection(self):
        script = (
            PROJECT_ROOT / "nnunet/queue_dataset503_preprocess_after_positive_20260827.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("waiting for completed positive-unlabeled public selector", script)
        self.assertIn('p.get("diagnostic_best")', script)
        self.assertIn("-d 503 -c 3d_fullres_ResEncS6", script)
        self.assertIn("-np 2", script)
        self.assertIn("gpu0<=1024 and gpu1<=1024", script)
        self.assertIn("organ_target_preprocessed_cases\":4", script)

    def test_safe_predictor_is_bound_to_organ_trainer(self):
        self.assertEqual(
            PREDICTOR_TRAINER_NAME,
            "nnUNetTrainerResEncS6PanCancerOrganTarget",
        )
        self.assertEqual(
            terminal_argument(["--terminal", "/tmp/organ.json"]),
            Path("/tmp/organ.json"),
        )
        with self.assertRaises(ValueError):
            terminal_argument([])

    def test_training_and_public_controllers_are_staged_and_hash_bound(self):
        training = (
            PROJECT_ROOT / "nnunet/run_resenc_s6_organ_target_2gpu_20260827.sh"
        ).read_text(encoding="utf-8")
        public = (
            PROJECT_ROOT / "nnunet/run_resenc_s6_organ_public_endpoints_20260827.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("stage_end_epochs=(10 25)", training)
        self.assertIn('export RESENC_ORGAN_STAGE_END_EPOCH="${stage_end}"', training)
        self.assertIn('p.get("selected") if isinstance(p.get("selected"),dict)', training)
        self.assertIn('row.get("checkpoint",{}).get("sha256")!=sha(parent_path)', training)
        self.assertIn("ENDPOINT_TERMINAL.json", training)
        self.assertIn('p.get("checkpoint",{}).get("sha256")!=sha', training)
        self.assertIn("epochs=(10 25)", public)
        self.assertIn('[[ "${epoch}" -eq 25 ]] && checkpoint_name="checkpoint_final.pth"', public)
        self.assertIn("--direct-compact-probabilities --sequential", public)
        self.assertNotIn("codabench", public.lower())
        self.assertNotIn("curl", public.lower())
        queue = (
            PROJECT_ROOT / "nnunet/queue_resenc_s6_organ_target_after_preprocess_20260827.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("waiting for Dataset503 preprocessing and positive selector", queue)
        self.assertIn("gpu0<=3072 and gpu1<=3072", queue)
        self.assertIn("util0<=5 and util1<=5", queue)
        self.assertIn("miccai-resenc-s6-organ-target-20260827", queue)


if __name__ == "__main__":
    unittest.main()
