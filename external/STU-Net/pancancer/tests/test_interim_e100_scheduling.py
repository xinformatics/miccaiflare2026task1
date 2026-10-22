import unittest
from pathlib import Path


PANCANCER = Path(__file__).resolve().parents[1]
GENERIC = (PANCANCER / "run_interim_e20_public_ensemble_20260825.sh").read_text(encoding="utf-8")
E100 = (PANCANCER / "run_interim_e100_public_ensemble_20260826.sh").read_text(encoding="utf-8")
RESUME_GENERIC = (PANCANCER / "resume_training_after_e20_ensemble_20260825.sh").read_text(encoding="utf-8")
RESUME_E100 = (PANCANCER / "resume_training_after_e100_ensemble_20260826.sh").read_text(encoding="utf-8")
SAVE_E100 = (PANCANCER / "save_interim_e100_qualified_ensemble_20260826.sh").read_text(encoding="utf-8")
DUAL = (PANCANCER / "launch_stunetb_dual_after_e100_20260826.sh").read_text(encoding="utf-8")
BASE = (PANCANCER / "continue_stunetb_base_gpu0_after_e100_20260826.sh").read_text(encoding="utf-8")
COMPONENT = (PANCANCER / "continue_stunetb_component_gpu1_after_e100_20260826.sh").read_text(encoding="utf-8")
SMOKE = (PANCANCER / "smoke_component_dataloader.py").read_text(encoding="utf-8")


class InterimE100SchedulingTest(unittest.TestCase):
    def test_wrapper_selects_exact_epoch_and_live_training_unit(self):
        self.assertIn("STUNET_INTERIM_EPOCH=100", E100)
        self.assertIn("STUNET_TRAINING_UNIT=miccai-stunetb-training-post-e20-20260825.service", E100)
        self.assertIn("stunetb_e100_ensemble_20260826", E100)

    def test_generic_controller_hash_binds_requested_optimizer_pair(self):
        self.assertIn('checkpoint_epoch_${epoch4}.pth', GENERIC)
        self.assertIn('int(x.get("current_epoch",-1))!=epoch', GENERIC)
        self.assertIn("torch.serialization.safe_globals", GENERIC)
        self.assertIn("--workers 1 --save-probabilities --dataset-role public", GENERIC)
        self.assertLess(GENERIC.index("stable=0"), GENERIC.index('systemctl --user stop'))

    def test_probability_shards_are_memory_safe_and_sequential(self):
        self.assertIn("for part_id in $(seq 0 49)", GENERIC)
        self.assertNotIn("shard_pids", GENERIC)
        self.assertIn("systemd-oomd", GENERIC)
        self.assertIn("--num-parts 50", GENERIC)
        self.assertNotIn("case_pids", GENERIC)
        self.assertIn('gpu=$((part_id % 2))', GENERIC)
        self.assertIn('analysis_workers="${STUNET_ANALYSIS_WORKERS:-4}"', GENERIC)
        self.assertIn('--workers "${analysis_workers}"', GENERIC)

    def test_resume_wrapper_preserves_same_run_and_epoch(self):
        self.assertIn("STUNET_INTERIM_EPOCH=100", RESUME_E100)
        self.assertIn("miccai-stunetb-dual-post-e100-20260826", RESUME_E100)
        self.assertIn("launch_stunetb_dual_after_e100_20260826.sh", RESUME_E100)
        self.assertIn("STUNET_POST_GATE_SCRIPT", RESUME_GENERIC)
        self.assertIn('terminal.get("predeclared_epochs")!=[epoch]', RESUME_GENERIC)
        self.assertIn('int(state.get("current_epoch",-1))!=epoch', RESUME_GENERIC)

    def test_dual_continuation_is_matched_and_gpu_disjoint(self):
        self.assertIn("miccai-stunetb-base-gpu0-post-e100-20260826", DUAL)
        self.assertIn("miccai-stunetb-component-gpu1-post-e100-20260826", DUAL)
        self.assertIn("CUDA_VISIBLE_DEVICES=0", BASE)
        self.assertIn("CUDA_VISIBLE_DEVICES=1", COMPONENT)
        for text in (BASE, COMPONENT):
            self.assertIn("checkpoint_epoch_0100.pth", text)
            self.assertIn("-num_gpus 1 --c", text)
            self.assertIn("global_batch_size", text)
        self.assertIn("STUNetTrainerPanCancerComponentBox", COMPONENT)
        self.assertIn("validate_post_e100_branch.py", BASE)
        self.assertIn("--mode base", BASE)
        self.assertIn("validate_post_e100_branch.py", COMPONENT)
        self.assertIn("--mode component", COMPONENT)
        validator = (PANCANCER / "validate_post_e100_branch.py").read_text(encoding="utf-8")
        self.assertIn("current_epoch", validator)
        self.assertIn("100 <= epoch < 300", validator)
        trainer = (PANCANCER / "STUNetTrainerPanCancer.py").read_text(encoding="utf-8")
        self.assertIn("ordered_round_robin", trainer)
        self.assertIn("augmentation_seed_base", trainer)

    def test_dual_launch_runs_export_equivalence_without_blocking_training(self):
        self.assertIn("validate_direct_compact_e100_real_case_20260826.sh", DUAL)
        self.assertIn('if ! /usr/bin/bash "${equivalence_script}"', DUAL)
        self.assertIn("continuing training with legacy export available", DUAL)

    def test_dual_training_scripts_do_not_read_validation_or_upload(self):
        for text in (DUAL, BASE, COMPONENT):
            lowered = text.lower()
            self.assertNotIn("validation-public", lowered)
            self.assertNotIn("validation-hidden", lowered)
            self.assertNotIn("codabench", lowered)
            self.assertNotIn(".zip", lowered)

    def test_real_data_smoke_asserts_matched_seeds_and_positive_target(self):
        self.assertIn("2_026_082_600", SMOKE)
        self.assertIn("2_026_092_600", SMOKE)
        self.assertIn('train_augmenter_parent', SMOKE)
        self.assertIn('positive_voxels_full', SMOKE)
        self.assertIn('public_read', SMOKE)
        self.assertNotIn("Validation-Public", SMOKE)
        self.assertNotIn("Validation-Hidden", SMOKE)

    def test_saver_wrapper_allows_only_epoch100(self):
        self.assertIn("STUNET_ALLOWED_EPOCHS=100", SAVE_E100)
        self.assertIn("stunetb_e100_ensemble_20260826", SAVE_E100)
        self.assertIn("save_public_qualified_ensemble_20260825.sh", SAVE_E100)

    def test_no_hidden_or_upload_in_public_wrapper(self):
        lowered = E100.lower()
        self.assertNotIn("validation-hidden", lowered)
        self.assertNotIn("codabench", lowered)
        self.assertNotIn(".zip", lowered)


if __name__ == "__main__":
    unittest.main()
