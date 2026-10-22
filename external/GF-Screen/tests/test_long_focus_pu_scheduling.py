import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = (
    ROOT / "run_long_focus_pu_after_direct_public_20260825.sh"
).read_text(encoding="utf-8")
EVALUATOR = (
    ROOT / "evaluate_long_focus_pu_champion_public_20260825.sh"
).read_text(encoding="utf-8")
TRAJECTORY = (
    ROOT / "evaluate_long_focus_pu_trajectory_public_20260825.sh"
).read_text(encoding="utf-8")
SELECTOR = (
    ROOT / "select_long_focus_pu_trajectory.py"
).read_text(encoding="utf-8")
TRAIN = (ROOT / "GF_RL_train.py").read_text(encoding="utf-8")
TRAINER = (ROOT / "GF_RL_trainer.py").read_text(encoding="utf-8")
VALIDATOR = (
    ROOT / "validate_positive_unlabeled_manifest.py"
).read_text(encoding="utf-8")


class LongFocusPositiveUnlabeledSchedulingTests(unittest.TestCase):
    def test_is_one_materially_long_focus_run(self):
        self.assertIn("gf_focus_pu_long30_e283_to313_20260825", CONTROLLER)
        self.assertIn("--start_epoch_override 283", CONTROLLER)
        self.assertIn("--max_epochs 313", CONTROLLER)
        self.assertIn("--samples_per_epoch 8192", CONTROLLER)
        self.assertIn("--distributed", CONTROLLER)
        self.assertIn("--component_uniform_crop", CONTROLLER)
        self.assertIn("--component_boxes", CONTROLLER)
        self.assertNotIn("train_arm", CONTROLLER)

    def test_uses_positive_only_train_unlabel_without_complete_cohorts(self):
        self.assertIn("--pseudo_abdomen_json", CONTROLLER)
        self.assertIn('item.get("pseudo_label") is not True', VALIDATOR)
        self.assertIn("validate_positive_unlabeled_manifest.py", CONTROLLER)
        self.assertIn('contract.get("excluded_images", -1)) != 6', CONTROLLER)
        self.assertIn("all masks partial; primary lesion only", CONTROLLER)
        self.assertIn("jsons/local/no_proxy_healthy.json", CONTROLLER)
        self.assertNotIn("--complete_label_cohorts_json", CONTROLLER)
        self.assertNotIn("verified7", CONTROLLER)

    def test_waits_for_complete_verified_pseudo_preprocessing_cache(self):
        self.assertIn("PSEUDO_CACHE_COMPLETE.json", CONTROLLER)
        self.assertIn("waiting for deterministic Train-Unlabel pseudo cache", CONTROLLER)
        self.assertIn("preprocess_positive_unlabeled_cache.py", CONTROLLER)
        self.assertIn("--verify-only", CONTROLLER)

    def test_resume_keeps_epoch283_teacher_immutable(self):
        self.assertIn(
            '--partial_label_teacher_checkpoint "${source_checkpoint}"',
            CONTROLLER,
        )
        self.assertIn("Frozen Focus teacher loaded from", TRAIN)
        self.assertIn("module.segmentation_model.", TRAIN)

    def test_failed_controller_restart_preserves_terminal_and_exact_log(self):
        self.assertIn('COMPLETE|SKIPPED) trap - EXIT; exit 0', CONTROLLER)
        self.assertIn('TERMINAL.failed_retry_', CONTROLLER)
        self.assertIn('preserved failed controller terminal for exact resume', CONTROLLER)
        self.assertIn('>> "${run_dir}/train.log" 2>&1', CONTROLLER)
        self.assertNotIn('> "${run_dir}/train.log" 2>&1', CONTROLLER.replace('>> "${run_dir}/train.log" 2>&1', ''))

    def test_wandb_is_stable_resume_safe_and_nonfatal(self):
        self.assertIn("stable_wandb_run_id", TRAINER)
        self.assertIn("safe_wandb_init", TRAINER)
        self.assertIn("safe_wandb_log", TRAINER)
        self.assertIn("safe_wandb_finish", TRAINER)
        self.assertNotIn("wandb.init(", TRAINER)
        self.assertNotIn("wandb_run.log(", TRAINER)
        self.assertIn('"wandb_run_id": hashlib.sha256(', CONTROLLER)
        self.assertIn("WANDB_INIT_TIMEOUT=60", CONTROLLER)

    def test_final_epoch_checkpoint_is_sufficient_for_evaluation(self):
        self.assertIn(
            '[[ -s "${run_dir}/checkpoint_epoch_0313.pt" ]]', CONTROLLER
        )
        self.assertNotIn(
            'checkpoint_epoch_0313.pt" && -s "${run_dir}/model_final.pt',
            CONTROLLER,
        )
        self.assertIn('LONG_FOCUS_RUN_DIR="${run_dir}"', CONTROLLER)
        self.assertIn("evaluate_long_focus_pu_trajectory_public_20260825.sh", CONTROLLER)

    def test_every_epoch_checkpoint_is_atomically_published(self):
        self.assertIn("def atomic_torch_save", TRAINER)
        self.assertIn("torch.save(payload, temporary)", TRAINER)
        self.assertIn("os.replace(temporary, destination)", TRAINER)
        self.assertIn("atomic_torch_save(save_dict, filename)", TRAINER)
        self.assertIn("atomic_copyfile(", TRAINER)

    def test_trajectory_predeclares_three_points_on_one_long_run(self):
        self.assertIn("epochs=(293 303 313)", TRAJECTORY)
        self.assertIn('"predeclared_epochs":[293,303,313]', TRAJECTORY)
        self.assertIn("evaluate_epoch 293 0", TRAJECTORY)
        self.assertIn("evaluate_epoch 303 1", TRAJECTORY)
        self.assertIn("evaluate_epoch 313 0", TRAJECTORY)
        self.assertIn("select_long_focus_pu_trajectory.py", TRAJECTORY)
        self.assertIn('selected = eligible[0] if eligible else None', SELECTOR)
        self.assertNotIn("GF_RL_train.py", TRAJECTORY)

    def test_waits_for_direct_route_and_preserves_prediction_priority(self):
        self.assertIn("focus_encoder_notta_direct_20260825/TERMINAL.json", CONTROLLER)
        self.assertIn("validate_direct_focus_public_terminal.py", CONTROLLER)
        self.assertIn('--route "${focus_route}"', CONTROLLER)
        self.assertIn('--calibration "${focus_calibration}"', CONTROLLER)
        self.assertIn('--summary "${focus_summary}"', CONTROLLER)
        self.assertIn("QUALIFY_SUBMISSION_PREPARATION", CONTROLLER)
        self.assertIn("QUALIFIED_ROUTE_NEEDS_PREDICTION", CONTROLLER)

    def test_public_screen_reproduces_champion_deployment(self):
        for token in (
            '--glance_threshold "${glance_threshold}"',
            "--lesion_threshold 0.50",
            "--min-mm3",
            "biomedparse_fusion.py",
            "screen_disconnected_guided_additions.py",
            "touching_component",
        ):
            self.assertIn(token, EVALUATOR)
        self.assertIn('glance_threshold="${LONG_FOCUS_GLANCE_THRESHOLD:-0.20}"', EVALUATOR)
        self.assertIn("deploy_at_minimum 500", EVALUATOR)
        self.assertIn("deploy_at_minimum 1000", EVALUATOR)
        self.assertIn("--min-dsc-gain 0.015", EVALUATOR)
        self.assertIn("--anchor-name gf_touchcomp_0825", EVALUATOR)

    def test_independent_deployment_branches_run_concurrently_and_join(self):
        launch500 = 'deploy_at_minimum 500 "${filtered500}" "${prompted500}" "${deployed500}" &'
        launch1000 = 'deploy_at_minimum 1000 "${filtered1000}" "${prompted1000}" "${deployed1000}" &'
        self.assertIn(launch500, EVALUATOR)
        self.assertIn(launch1000, EVALUATOR)
        self.assertIn("pid500=$!", EVALUATOR)
        self.assertIn("pid1000=$!", EVALUATOR)
        self.assertIn('wait "${pid500}" || deploy_status=$?', EVALUATOR)
        self.assertIn('wait "${pid1000}" || deploy_status=$?', EVALUATOR)
        self.assertLess(EVALUATOR.index(launch500), EVALUATOR.index(launch1000))
        self.assertLess(EVALUATOR.index(launch1000), EVALUATOR.index("touch_count="))

    def test_public_screen_cannot_read_hidden_or_package(self):
        lowered = (EVALUATOR + TRAJECTORY + SELECTOR).lower()
        self.assertNotIn("validation-hidden", lowered)
        self.assertNotIn("codabench", lowered)
        self.assertNotIn(".zip", lowered)
        self.assertIn('"hidden_read": false', lowered)
        self.assertIn('"submission_created": false', lowered)


if __name__ == "__main__":
    unittest.main()
