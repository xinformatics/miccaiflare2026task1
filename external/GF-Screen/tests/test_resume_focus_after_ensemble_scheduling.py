import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "resume_focus_after_ensemble_20260825.sh").read_text(
    encoding="utf-8"
)


class ResumeFocusAfterEnsembleSchedulingTests(unittest.TestCase):
    def test_waits_for_final_ensemble_save_decision(self):
        self.assertIn("stunetb_ensemble_saved_20260825/TERMINAL.json", SCRIPT)
        self.assertIn("COMPLETE|SKIPPED) break", SCRIPT)
        self.assertIn("replacement of failed STU-Net ensemble terminal", SCRIPT)

    def test_hash_binds_exact_checkpoint_303_resume_state(self):
        self.assertIn("TRAINING_DEFERRED_FOR_ENSEMBLE.json", SCRIPT)
        self.assertIn('item.get("sha256")!=sha(checkpoint)', SCRIPT)
        self.assertIn('int(state.get("epoch",-1))!=302', SCRIPT)
        self.assertIn('state.get("optimizer_seg")', SCRIPT)
        self.assertIn('state.get("scheduler_seg")', SCRIPT)

    def test_resumes_existing_long_focus_controller_without_submission_access(self):
        self.assertIn("run_long_focus_pu_after_direct_public_20260825.sh", SCRIPT)
        self.assertIn("miccai-long-focus-pu-resume-after-ensemble-20260825", SCRIPT)
        lowered = SCRIPT.lower()
        self.assertNotIn("codabench.org", lowered)
        self.assertNotIn("validation-hidden", lowered)
        self.assertNotIn(".zip", lowered)


if __name__ == "__main__":
    unittest.main()
