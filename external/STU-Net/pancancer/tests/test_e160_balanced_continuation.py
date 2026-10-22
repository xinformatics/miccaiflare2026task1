import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "run_component_e160_balanced_continuation_20260826.sh").read_text(
    encoding="utf-8"
)
TRAINER = (ROOT / "STUNetTrainerPanCancer.py").read_text(encoding="utf-8")
RUNNER = (ROOT / "run_pancancer_stunet.py").read_text(encoding="utf-8")


class E160BalancedContinuationTests(unittest.TestCase):
    def test_parent_is_exact_hidden_leading_epoch160_state(self):
        self.assertIn("checkpoint_inference_epoch_0160.pth", SCRIPT)
        self.assertIn(
            'parent_sha="81b9eed4105e89e8399c091984db4fb9ce1e37089d07a3fd4c032782f8d7cd74"',
            SCRIPT,
        )
        self.assertIn('-pretrained_weights "${parent}"', SCRIPT)
        self.assertIn("EXPECTED_COMPONENT_E160_SHA256", RUNNER)
        self.assertIn("component-box epoch-160 tensor inventory mismatch", RUNNER)

    def test_recipe_is_eighty_epoch_balanced_low_lr_continuation(self):
        self.assertIn("class STUNetTrainerPanCancerComponentBoxE160Balanced", TRAINER)
        self.assertIn("self.initial_lr = 3e-5", TRAINER)
        self.assertIn("self.num_epochs = 80", TRAINER)
        self.assertIn("self.save_every = 1", TRAINER)
        self.assertIn("PartialBoxTverskyCELoss(alpha=0.5, beta=0.5)", TRAINER)
        self.assertIn("optimizer_restart", TRAINER)

    def test_repurposes_only_base_gpu_and_restores_it_on_failure(self):
        self.assertIn('systemctl --user stop "${base_unit}"', SCRIPT)
        self.assertNotIn("component-e300-public", SCRIPT)
        self.assertIn("restore_base", SCRIPT)
        self.assertLess(SCRIPT.index("restore_base\n"), SCRIPT.index('exit "${status}"'))
        self.assertIn("CUDA_VISIBLE_DEVICES=0", SCRIPT)

    def test_oom_safe_resume_preserves_optimizer_and_compiled_execution(self):
        self.assertIn('latest="${fold}/checkpoint_latest.pth"', SCRIPT)
        self.assertIn('launch_args=(-pretrained_weights "${parent}")', SCRIPT)
        self.assertIn('launch_args=(--c)', SCRIPT)
        self.assertIn('payload.get("optimizer_state")', SCRIPT)
        self.assertIn('export nnUNet_n_proc_DA="${BALANCED_N_PROC_DA:-1}"', SCRIPT)
        self.assertIn('export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"', SCRIPT)
        self.assertIn("export nnUNet_compile=false", SCRIPT)
        self.assertIn('Path("/proc/pressure/memory")', SCRIPT)
        self.assertIn("int(sys.argv[1]) >= 20*1024*1024", SCRIPT)
        self.assertIn("float(sys.argv[2]) <= 5.0", SCRIPT)
        self.assertIn("waiting for host memory recovery", SCRIPT)

    def test_training_is_label_role_safe_and_cannot_submit(self):
        self.assertNotIn("Validation-Public", SCRIPT)
        self.assertNotIn("Validation-Hidden", SCRIPT)
        self.assertNotIn("codabench", SCRIPT.lower())
        self.assertNotIn("zip -", SCRIPT)
        self.assertIn(
            '"public_read":False,"hidden_read":False,"submission_created":False,"uploaded":False',
            SCRIPT,
        )


if __name__ == "__main__":
    unittest.main()
