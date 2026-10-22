import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "run_training_after_independent_audit_20260825.sh"


class STUNetTrainingSchedulingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_waits_for_independent_public_audit_and_idle_pair(self):
        self.assertIn("independent_baseline_saved_20260825/TERMINAL.json", self.text)
        self.assertIn("COMPLETE|SKIPPED) break", self.text)
        self.assertIn("wait_for_idle_pair", self.text)
        self.assertIn("nvidia-smi --id=0", self.text)
        self.assertIn("nvidia-smi --id=1", self.text)

    def test_smoke_precedes_full_two_gpu_training(self):
        smoke = self.text.index("-tr STUNetTrainerPanCancerSmoke")
        full = self.text.index("-tr STUNetTrainerPanCancer -p nnUNetPlans")
        self.assertLess(smoke, full)
        self.assertGreaterEqual(self.text.count("-num_gpus 2"), 3)

    def test_binds_repo_checkpoint_plan_and_recipe(self):
        self.assertIn("0652eaf5cafda6bc269942238c3695994f2dc4f0", self.text)
        self.assertIn("7232a6296d320da1d60debe873a1d3fcad74e9404272db8f7440e30d62c5997a", self.text)
        self.assertIn("3d_fullres_STUNetB_PartialBox", self.text)
        self.assertIn('"epochs":300', self.text)
        self.assertIn('"bbox_context_mm":16', self.text)
        self.assertIn("export nnUNet_n_proc_DA=8", self.text)
        self.assertIn('"augmentation_workers":16', self.text)

    def test_training_cannot_touch_public_hidden_or_submission(self):
        lowered = self.text.lower()
        self.assertNotIn("validation-public", lowered)
        self.assertNotIn("validation-hidden", lowered)
        self.assertNotIn("codabench", lowered)
        self.assertNotIn("zip -", lowered)
        self.assertIn('"public_read":False', self.text)
        self.assertIn('"hidden_read":False', self.text)
        self.assertIn('"submission_created":False', self.text)


if __name__ == "__main__":
    unittest.main()
