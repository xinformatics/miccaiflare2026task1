import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "run_focus_routed_tta_completed_folds_accelerator_20260824.sh"
).read_text(encoding="utf-8")


class FocusRoutedTTAEarlyAcceleratorTests(unittest.TestCase):
    def test_reuses_fold4_and_only_runs_completed_fold01(self):
        self.assertIn("for fold in 0 1; do", SCRIPT)
        self.assertIn("for fold in 0 1 4; do", SCRIPT)
        self.assertIn("source_prediction=", SCRIPT)
        self.assertIn("ln -s", SCRIPT)
        self.assertIn("evaluate_focus_checkpoint_tta.py", SCRIPT)

    def test_is_pilot_gated_guarded_and_local_only(self):
        self.assertIn("PROMOTE_TTA_TO_FIVEFOLD", SCRIPT)
        self.assertIn('"${used}" -le 4096', SCRIPT)
        self.assertNotIn("val_image", SCRIPT)
        self.assertNotIn("submission.zip", SCRIPT)
        self.assertNotIn("codabench", SCRIPT.lower())


if __name__ == "__main__":
    unittest.main()
