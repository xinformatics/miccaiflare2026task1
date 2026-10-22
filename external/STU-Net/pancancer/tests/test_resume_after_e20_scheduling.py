import unittest
from pathlib import Path


PANCANCER = Path(__file__).resolve().parents[1]
SCRIPT = (PANCANCER / "resume_training_after_e20_ensemble_20260825.sh").read_text(encoding="utf-8")


class ResumeAfterE20SchedulingTest(unittest.TestCase):
    def test_waits_for_final_saver_decision(self):
        self.assertIn('interim_epoch="${STUNET_INTERIM_EPOCH:-20}"', SCRIPT)
        self.assertIn('stunetb_e${interim_epoch}_ensemble_saved_20260825', SCRIPT)
        self.assertIn("COMPLETE|SKIPPED) break", SCRIPT)
        self.assertIn("replacement of failed epoch-%s saver terminal", SCRIPT)

    def test_verifies_both_epoch20_training_states(self):
        self.assertIn('checkpoint_epoch_${epoch4}.pth', SCRIPT)
        self.assertIn("checkpoint_latest.pth", SCRIPT)
        self.assertIn('int(state.get("current_epoch",-1))!=epoch', SCRIPT)
        self.assertIn("torch.serialization.safe_globals", SCRIPT)
        self.assertIn('item.get("sha256")!=sha(path)', SCRIPT)

    def test_restarts_same_full_training_recipe_only(self):
        self.assertIn("run_training_after_independent_audit_20260825.sh", SCRIPT)
        lowered=SCRIPT.lower()
        self.assertNotIn("validation-public", lowered)
        self.assertNotIn("validation-hidden", lowered)
        self.assertNotIn(".zip", lowered)
        self.assertNotIn("codabench", lowered)


if __name__ == "__main__":
    unittest.main()
