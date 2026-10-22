import json
import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from nnunet.resenc_s6_partialbox import (  # noqa: E402
    nnUNetTrainerResEncS6PanCancerPartialBox,
)
from nnunet.run_resenc_s6_partialbox import resolve_trainer  # noqa: E402
from STUNetTrainerPanCancer import matched_augmentation_seed_base  # noqa: E402


class ResEncS6PartialBoxTests(unittest.TestCase):
    def test_runner_resolves_exact_workspace_trainer(self):
        self.assertIs(
            resolve_trainer(None, "nnUNetTrainerResEncS6PanCancerPartialBox", None),
            nnUNetTrainerResEncS6PanCancerPartialBox,
        )

    def test_recipe_is_long_from_scratch_and_partial_label_safe(self):
        trainer = object.__new__(nnUNetTrainerResEncS6PanCancerPartialBox)
        trainer.enable_deep_supervision = False
        loss = trainer._build_loss()
        self.assertEqual(loss.alpha, 0.25)
        self.assertEqual(loss.beta, 0.75)
        self.assertEqual(trainer.expected_case_count, 17_575)
        self.assertEqual(trainer.expected_source_count, 28)
        self.assertEqual(trainer.expected_parameter_count, 107_467_050)
        self.assertEqual(trainer.expected_global_batch_size, 16)
        self.assertEqual(trainer.component_center_fraction, 0.25)
        self.assertEqual(trainer.landmark_epochs, (100, 200, 300, 400))
        self.assertEqual(trainer.stage_end_epochs, (100, 200, 300, 400, 500))
        self.assertEqual(trainer.training_horizon_epochs, 500)

    def test_optimizer_scheduler_keeps_500_epoch_horizon_between_stages(self):
        trainer = object.__new__(nnUNetTrainerResEncS6PanCancerPartialBox)
        trainer.network = torch.nn.Linear(2, 2)
        trainer.initial_lr = 1e-3
        trainer.weight_decay = 3e-5
        trainer.lr_schedule_horizon_epochs = 500
        optimizer, scheduler = trainer.configure_optimizers()
        self.assertEqual(optimizer.defaults["momentum"], 0.99)
        self.assertTrue(optimizer.defaults["nesterov"])
        self.assertEqual(scheduler.max_steps, 500)
        scheduler.step(100)
        expected = 1e-3 * (1.0 - 100 / 500) ** 0.9
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], expected)

    def test_plan_is_exact_six_stage_107m_recipe_and_reuses_cache(self):
        plan_path = (
            PROJECT_ROOT
            / "nnUNet_preprocessed"
            / "Dataset501_PanCancer"
            / "nnUNetPlans.json"
        )
        plans = json.loads(plan_path.read_text(encoding="utf-8"))
        configuration = plans["configurations"]["3d_fullres_ResEncS6"]
        architecture = configuration["architecture"]
        kwargs = architecture["arch_kwargs"]
        self.assertTrue(
            architecture["network_class_name"].endswith("ResidualEncoderUNet")
        )
        self.assertEqual(kwargs["n_stages"], 6)
        self.assertEqual(kwargs["features_per_stage"], [32, 64, 128, 256, 320, 320])
        self.assertEqual(kwargs["n_blocks_per_stage"], [1, 3, 4, 6, 6, 6])
        self.assertEqual(configuration["patch_size"], [128, 160, 160])
        self.assertEqual(configuration["spacing"], [2.4, 1.6, 1.6])
        self.assertEqual(
            configuration["data_identifier"],
            "nnUNetPlans_3d_fullres_S4D2W32",
        )

    def test_target_conversion_is_component_box_not_global_box(self):
        source = Path(
            sys.modules[nnUNetTrainerResEncS6PanCancerPartialBox.__module__].__file__
        ).read_text(encoding="utf-8")
        self.assertIn("STUNetTrainerPanCancerComponentBoxE160SourceStratified", source)
        self.assertNotIn("training_complete", source)
        self.assertNotIn("public_labels", source)
        self.assertNotIn("hidden_labels", source)

    def test_public_policy_is_frozen_to_five_endpoints_and_four_deployments(self):
        policy = json.loads(
            (
                PROJECT_ROOT
                / "nnunet"
                / "RESENC_S6_PUBLIC_SELECTION_POLICY_20260827.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(policy["status"], "PREDECLARED_BEFORE_TRAINING")
        self.assertEqual(policy["checkpoint_epochs"], [100, 200, 300, 400, 500])
        self.assertEqual(
            [item["name"] for item in policy["deployments"]],
            [
                "resenc_standalone_p50",
                "gf_anchor_confirm_either_p70",
                "gf_resenc_boundary3mm_p70p50_ftu",
                "gf_union_resenc_p40_ftu_s10_mm3000",
            ],
        )
        qualification = policy["selection"]["qualification"]
        self.assertEqual(
            qualification["minimum_fixed_public_DSC"], 0.6366639257651584
        )
        self.assertEqual(
            qualification["minimum_fixed_public_NSD"], 0.5347520263176905
        )
        self.assertEqual(qualification["minimum_DSC_gain_over_incumbent"], 0.015)
        self.assertFalse(policy["threshold_search_after_public_read"])

    def test_training_controller_requires_real_data_target_audit(self):
        controller = (
            PROJECT_ROOT / "nnunet/run_resenc_s6_partialbox_2gpu_20260827.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("RESENC_S6_PARTIALBOX_REALDATA_AUDIT_20260827.json", controller)
        self.assertIn("KEEP_16MM_COMPONENT_BOX_AND_TVERSKY_A025_B075", controller)
        self.assertIn("24.67456449552064", controller)

    def test_training_releases_gpus_at_each_hash_bound_public_endpoint(self):
        controller = (
            PROJECT_ROOT / "nnunet/run_resenc_s6_partialbox_2gpu_20260827.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("stage_end_epochs=(100 200 300 400 500)", controller)
        self.assertIn('export RESENC_STAGE_END_EPOCH="${stage_end}"', controller)
        self.assertIn(
            'export PANCANCER_AUGMENTATION_SEED_OFFSET="$((current * 1000000))"',
            controller,
        )
        self.assertIn("ENDPOINT_TERMINAL.json", controller)
        self.assertIn('p.get("checkpoint",{}).get("sha256")!=sha', controller)
        self.assertIn('p.get("dataset_role")!="public"', controller)
        self.assertIn('for candidate in "${final}" "${latest}"', controller)
        self.assertIn("candidate_epoch > current", controller)

    def test_staged_processes_do_not_repeat_the_same_augmentation_stream(self):
        parent = (
            PROJECT_ROOT
            / "external/STU-Net/pancancer/STUNetTrainerPanCancer.py"
        ).read_text(encoding="utf-8")
        self.assertIn("PANCANCER_AUGMENTATION_SEED_OFFSET", parent)
        self.assertIn('getattr(self, "local_rank", 0)', parent)
        self.assertIn("must be nonnegative", parent)

    def test_stage_rank_and_loader_seed_namespaces_are_disjoint(self):
        rank0_train = matched_augmentation_seed_base(0, 0, 0)
        rank1_train = matched_augmentation_seed_base(0, 1, 0)
        rank0_validation = matched_augmentation_seed_base(0, 0, 1)
        stage100_rank0 = matched_augmentation_seed_base(100_000_000, 0, 0)
        self.assertEqual(rank1_train - rank0_train, 100_000)
        self.assertEqual(rank0_validation - rank0_train, 10_000)
        self.assertEqual(stage100_rank0 - rank0_train, 100_000_000)
        self.assertEqual(len({rank0_train, rank1_train, rank0_validation, stage100_rank0}), 4)
        with self.assertRaisesRegex(ValueError, "must be nonnegative"):
            matched_augmentation_seed_base(-1, 0, 0)


if __name__ == "__main__":
    unittest.main()
