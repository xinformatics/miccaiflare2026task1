import unittest
from pathlib import Path


SCRIPT = (Path(__file__).resolve().parents[1] / "run_matched_e130_public_20260826.sh").read_text(encoding="utf-8")
SAVER = (Path(__file__).resolve().parents[1] / "save_matched_e130_qualified_20260826.sh").read_text(encoding="utf-8")
RESUME = (Path(__file__).resolve().parents[1] / "resume_dual_after_matched_e130_20260826.sh").read_text(encoding="utf-8")
GENERIC_SAVER = (Path(__file__).resolve().parents[1] / "save_public_qualified_ensemble_20260825.sh").read_text(encoding="utf-8")


class MatchedE130SchedulingTest(unittest.TestCase):
    def test_exact_matched_epoch_and_disjoint_models(self):
        self.assertIn("checkpoint_epoch_0130.pth", SCRIPT)
        self.assertIn("STUNetTrainerPanCancer__", SCRIPT)
        self.assertIn("STUNetTrainerPanCancerComponentBox__", SCRIPT)
        self.assertIn("--mode base", SCRIPT)
        self.assertIn("--mode component", SCRIPT)

    def test_epoch_check_reuses_safe_validator_output(self):
        self.assertIn('base_checkpoint_validation=$("${python_bin}" "${validator}"', SCRIPT)
        self.assertIn('component_checkpoint_validation=$("${python_bin}" "${validator}"', SCRIPT)
        self.assertIn('json.loads(sys.argv[1])["latest_epoch"]', SCRIPT)
        self.assertNotIn("torch.load(sys.argv[1]", SCRIPT)

    def test_both_training_services_pause_and_resume(self):
        self.assertIn("miccai-stunetb-base-gpu0-post-e100-20260826.service", SCRIPT)
        self.assertIn("miccai-stunetb-component-gpu1-post-e100-20260826.service", SCRIPT)
        self.assertIn("systemctl --user stop", SCRIPT)
        self.assertIn("launch_stunetb_dual_after_e100_20260826.sh", SCRIPT)
        self.assertIn("stunetb_matched_e130_saved_20260826", RESUME)
        self.assertIn("launch_stunetb_dual_after_e100_20260826.sh", RESUME)

    def test_inference_uses_disjoint_gpus_but_is_host_memory_safe(self):
        self.assertIn("predict_casewise", SCRIPT)
        self.assertIn("for part_id in $(seq 0 49)", SCRIPT)
        self.assertIn('--gpu "${gpu}" --part-id "${part_id}" --num-parts 50', SCRIPT)
        self.assertIn("--save-probabilities", SCRIPT)
        self.assertNotIn("base_pid", SCRIPT)
        self.assertNotIn("component_pid", SCRIPT)
        self.assertNotIn("case_pids", SCRIPT)
        self.assertIn('gpu=$((part_id % 2))', SCRIPT)
        self.assertIn('analysis_workers="${STUNET_ANALYSIS_WORKERS:-4}"', SCRIPT)
        self.assertIn('OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"', SCRIPT)
        self.assertIn("direct_compact_equivalence_e100_20260826/EQUIVALENCE.json", SCRIPT)
        self.assertIn('x.get("maximum_probability_code_difference")!=0', SCRIPT)
        self.assertIn('"${compact_args[@]}"', SCRIPT)

    def test_selection_requires_both_immutable_gates(self):
        self.assertIn('row["Lesion_DSC"]>=0.6154316379500749', SCRIPT)
        self.assertIn('row["Lesion_NSD"]>=0.5104225107769019', SCRIPT)
        self.assertIn("maximum fixed-public DSC subject to immutable DSC/NSD gates", SCRIPT)

    def test_no_hidden_archive_upload_or_codabench_path(self):
        lowered = SCRIPT.lower()
        self.assertNotIn("validation-hidden", lowered)
        self.assertNotIn(".zip", lowered)
        self.assertNotIn("codabench.org", lowered)
        self.assertNotIn("requests.post", lowered)
        self.assertIn('"hidden_read":False', SCRIPT)
        self.assertIn('"uploaded":False', SCRIPT)

    def test_qualified_saver_supports_either_selected_branch(self):
        self.assertIn("STUNET_ALLOWED_EPOCHS=130", SAVER)
        self.assertIn("stunetb_matched_e130_20260826", SAVER)
        self.assertIn('branch=public.get("selected_branch","base_global_box")', GENERIC_SAVER)
        self.assertIn('elif branch=="component_box"', GENERIC_SAVER)
        self.assertIn("STUNetTrainerPanCancerComponentBox__", GENERIC_SAVER)
        self.assertIn('"uploaded":False', GENERIC_SAVER)
        self.assertIn("direct_compact_equivalence_e100_20260826/EQUIVALENCE.json", GENERIC_SAVER)
        self.assertIn('"${compact_args[@]}"', GENERIC_SAVER)


if __name__ == "__main__":
    unittest.main()
