import unittest
from pathlib import Path


SCRIPT = (Path(__file__).resolve().parents[1] / "resume_balanced_after_e40_interim_20260826.sh").read_text()


class ResumeBalancedAfterE40Tests(unittest.TestCase):
    def test_resume_is_bound_to_exact_checkpoint_and_safe_resume_modes(self):
        self.assertIn("3015cdf2d9b267989fd2fde2939cdac8670c55c31b96fd50f59eadf9da2a29f6", SCRIPT)
        self.assertIn("trainer_pid=2087594", SCRIPT)
        self.assertIn("wandb_pid=2087741", SCRIPT)
        self.assertIn('[[ "${state}" == T || "${state}" == t ]]', SCRIPT)
        self.assertIn('kill -CONT -- "-${trainer_pid}" "-${wandb_pid}"', SCRIPT)
        self.assertIn('payload.get("current_epoch")!=47', SCRIPT)
        self.assertIn('payload.get("trainer_name")!="STUNetTrainerPanCancerComponentBoxE160Balanced"', SCRIPT)
        self.assertIn('systemctl --user start "${trainer_unit}"', SCRIPT)
        self.assertIn('optimizer_checkpoint_current_epoch_47', SCRIPT)
        self.assertIn('map(Path,sys.argv[1:4])', SCRIPT)

    def test_resume_requires_valid_interim_and_restores_final_evaluator(self):
        self.assertIn("INTERIM_E40_PUBLIC_READY_TRAINING_MAY_RESUME", SCRIPT)
        self.assertIn("BALANCED_TRAINING_RESUMED_AFTER_E40_PUBLIC", SCRIPT)
        self.assertIn("miccai-stunetb-balanced-e40-e80-public-20260826.service", SCRIPT)
        self.assertNotIn("Validation-Hidden", SCRIPT)
        self.assertNotIn("codabench.org", SCRIPT)


if __name__ == "__main__":
    unittest.main()
