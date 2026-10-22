import json
import importlib
import os
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("nnUNet_raw", str(PROJECT_ROOT / "nnUNet_raw"))
os.environ.setdefault(
    "nnUNet_preprocessed", str(PROJECT_ROOT / "nnUNet_preprocessed")
)
os.environ.setdefault("nnUNet_results", str(PROJECT_ROOT / "nnUNet_results"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-resenc-positive-tests")
sys.path.insert(0, str(PROJECT_ROOT))

from nnunet.resenc_s6_positive_unlabeled import (  # noqa: E402
    EXPECTED_SORTED_IDENTIFIER_SHA256,
    PSEUDO_EXPOSURE,
    nnUNetTrainerResEncS6PanCancerPositiveUnlabeled,
    positive_unlabeled_loader_factory,
    positive_unlabeled_sampling_probabilities,
)
from nnunet.run_resenc_s6_positive_unlabeled import (  # noqa: E402
    resolve_trainer,
)
from nnunet.select_resenc_s6_positive_public import (  # noqa: E402
    load_policy,
)
from nnunet import select_resenc_s6_public as base_public_selector  # noqa: E402
from STUNetTrainerPanCancerSourceBalanced import (  # noqa: E402
    identifier_inventory_sha256,
)


class FakeDataset:
    def __init__(self, identifiers, foreground_counts=None):
        self.identifiers = tuple(identifiers)
        self.foreground_counts = foreground_counts


class FakeLoader:
    def __init__(self, data, *args, sampling_probabilities=None, **kwargs):
        del args, kwargs
        self._data = data
        self.sampling_probabilities = sampling_probabilities


class ResEncS6PositiveUnlabeledTests(unittest.TestCase):
    def test_runner_resolves_exact_workspace_trainer(self):
        self.assertIs(
            resolve_trainer(
                None,
                "nnUNetTrainerResEncS6PanCancerPositiveUnlabeled",
                None,
            ),
            nnUNetTrainerResEncS6PanCancerPositiveUnlabeled,
        )

    def test_fixed_pseudo_exposure_and_labeled_source_mixture(self):
        identifiers = (
            "Liver-A_001",
            "Liver-A_002",
            "Kidney-B_001",
            "Pseudo-AMOS-2350_0001",
            "Pseudo-MSD-506_0001",
        )
        probabilities = positive_unlabeled_sampling_probabilities(
            identifiers, [50, 500, 10_000, 100, 100]
        )
        pseudo = np.asarray([item.startswith("Pseudo-") for item in identifiers])
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertAlmostEqual(float(probabilities[pseudo].sum()), PSEUDO_EXPOSURE)
        self.assertAlmostEqual(float(probabilities[~pseudo].sum()), 1 - PSEUDO_EXPOSURE)
        self.assertTrue(np.all(probabilities > 0))
        self.assertAlmostEqual(float(probabilities[-1]), float(probabilities[-2]))

    def test_sampler_rejects_unsafe_partitions_and_bad_alignment(self):
        with self.assertRaises(ValueError):
            positive_unlabeled_sampling_probabilities(["Liver_001"], [10])
        with self.assertRaises(ValueError):
            positive_unlabeled_sampling_probabilities(
                ["Pseudo-AMOS-2350_0001"], [10]
            )
        with self.assertRaises(ValueError):
            positive_unlabeled_sampling_probabilities(
                ["Liver_001", "Pseudo-AMOS-2350_0001"], [10]
            )

    def test_loader_factory_changes_training_only(self):
        resolver = lambda data: np.asarray(data.foreground_counts)
        factory = positive_unlabeled_loader_factory(FakeLoader, 0.25, resolver)
        training = factory(
            FakeDataset(
                ["Liver_001", "Kidney_001", "Pseudo-AMOS-2350_0001"],
                [100, 1_000, 200],
            ),
            sampling_probabilities=None,
        )
        validation = factory(
            FakeDataset(["Liver_002"], [100]), sampling_probabilities=None
        )
        self.assertIsInstance(training, FakeLoader)
        self.assertAlmostEqual(
            float(training.sampling_probabilities[-1]), PSEUDO_EXPOSURE
        )
        self.assertEqual(training.lesion_uniform_center_fraction, 0.25)
        self.assertIsNone(validation.sampling_probabilities)

    def test_dataset_identity_inventory_and_output_are_isolated(self):
        raw = PROJECT_ROOT / "nnUNet_raw/Dataset502_PanCancerPositiveUnlabeled"
        preprocessed = (
            PROJECT_ROOT
            / "nnUNet_preprocessed/Dataset502_PanCancerPositiveUnlabeled"
        )
        identifiers = sorted(
            path.name.removesuffix("_0000.nii.gz")
            for path in (raw / "imagesTr").glob("*_0000.nii.gz")
        )
        self.assertEqual(len(identifiers), 19_886)
        self.assertEqual(
            identifier_inventory_sha256(identifiers),
            EXPECTED_SORTED_IDENTIFIER_SHA256,
        )
        plans = json.loads((preprocessed / "nnUNetPlans.json").read_text())
        dataset = json.loads((preprocessed / "dataset.json").read_text())
        self.assertEqual(plans["dataset_name"], "Dataset502_PanCancerPositiveUnlabeled")
        self.assertFalse((preprocessed / "nnUNetPlans.json").is_symlink())
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
            trainer = nnUNetTrainerResEncS6PanCancerPositiveUnlabeled(
                plans,
                "3d_fullres_ResEncS6",
                "all",
                dataset,
                device=torch.device("cpu"),
            )
        self.assertEqual(trainer.initial_lr, 1e-4)
        self.assertEqual(trainer.num_epochs, 100)
        self.assertEqual(trainer.positive_stage_end_epochs, (25, 50, 100))
        self.assertEqual(trainer.lr_schedule_horizon_epochs, 100)
        self.assertEqual(trainer.configuration_manager.batch_size, 16)
        self.assertIn(
            "Dataset502_PanCancerPositiveUnlabeled", trainer.output_folder
        )

    def test_public_policy_is_frozen_before_training(self):
        policy = load_policy(
            PROJECT_ROOT
            / "nnunet/RESENC_S6_POSITIVE_UNLABELED_PUBLIC_SELECTION_POLICY_20260827.json"
        )
        self.assertEqual(policy["checkpoint_epochs"], [25, 50, 100])
        self.assertEqual(
            [item["name"] for item in policy["deployments"]],
            [
                "resenc_standalone_p50",
                "gf_anchor_confirm_either_p70",
                "gf_resenc_boundary3mm_p70p50_ftu",
                "gf_union_resenc_p40_ftu_s10_mm3000",
            ],
        )
        self.assertFalse(policy["threshold_search_after_public_read"])
        self.assertEqual(
            policy["selection"]["qualification"]["minimum_fixed_public_DSC"],
            0.6366639257651584,
        )

    def test_fresh_positive_optimizer_uses_its_100_epoch_schedule(self):
        trainer = object.__new__(
            nnUNetTrainerResEncS6PanCancerPositiveUnlabeled
        )
        trainer.network = torch.nn.Linear(2, 2)
        trainer.initial_lr = 1e-4
        trainer.weight_decay = 3e-5
        trainer.lr_schedule_horizon_epochs = 100
        optimizer, scheduler = trainer.configure_optimizers()
        self.assertEqual(scheduler.max_steps, 100)
        scheduler.step(50)
        expected = 1e-4 * (1.0 - 50 / 100) ** 0.9
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], expected)

    def test_positive_selector_does_not_mutate_base_policy_loader(self):
        base_policy = load_policy.__module__.replace(
            "select_resenc_s6_positive_public", "select_resenc_s6_public"
        )
        self.assertEqual(
            base_public_selector.load_policy.__module__,
            base_policy,
        )
        payload = base_public_selector.load_policy(
            PROJECT_ROOT / "nnunet/RESENC_S6_PUBLIC_SELECTION_POLICY_20260827.json"
        )
        self.assertEqual(payload["checkpoint_epochs"], [100, 200, 300, 400, 500])

    def test_positive_training_and_public_evaluation_are_staged_and_hash_bound(self):
        training = (
            PROJECT_ROOT
            / "nnunet/run_resenc_s6_positive_unlabeled_2gpu_20260827.sh"
        ).read_text(encoding="utf-8")
        public = (
            PROJECT_ROOT
            / "nnunet/run_resenc_s6_positive_public_endpoints_20260827.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("stage_end_epochs=(25 50 100)", training)
        self.assertIn('export RESENC_POSITIVE_STAGE_END_EPOCH="${stage_end}"', training)
        self.assertIn("ENDPOINT_TERMINAL.json", training)
        self.assertIn('p.get("checkpoint",{}).get("sha256")!=sha', training)
        self.assertIn('for candidate in "${final}" "${latest}"', training)
        self.assertIn("candidate_epoch > current", training)
        self.assertNotIn(
            "waiting for positive-unlabeled ResEnc training terminal", public
        )
        self.assertIn("waiting for positive ResEnc endpoint checkpoint", public)
        self.assertIn("ENDPOINT_TERMINAL.json", public)
        self.assertIn("hashlib.sha256(analysis.read_bytes())", public)


if __name__ == "__main__":
    unittest.main()
