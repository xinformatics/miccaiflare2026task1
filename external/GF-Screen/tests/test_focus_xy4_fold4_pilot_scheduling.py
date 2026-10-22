import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "run_focus_xy4_fold4_pilot_20260824.sh").read_text(
    encoding="utf-8"
)


class FocusXY4Fold4PilotSchedulingTests(unittest.TestCase):
    def test_waits_for_three_view_fivefold_and_is_local_only(self):
        self.assertIn("routed_tta_fivefold_20260824/TERMINAL.json", SCRIPT)
        self.assertIn("evaluate_focus_checkpoint_xy4.py", SCRIPT)
        self.assertIn("select_focus_xy4_pilot.py", SCRIPT)
        self.assertIn("audit_focus_fold_epoch_sources.py", SCRIPT)
        self.assertNotIn("val_image", SCRIPT)
        self.assertNotIn("submission.zip", SCRIPT)
        self.assertNotIn("codabench", SCRIPT.lower())

    def test_runs_only_frozen_source_epochs_on_guarded_gpus(self):
        self.assertIn('(\"weak_sources\",\"lung_cells\")', SCRIPT)
        self.assertIn('"${used}" -le 4096', SCRIPT)
        self.assertIn('run_one "${gpu}" "${specification}"', SCRIPT)
        self.assertIn("gpu=$((1-gpu))", SCRIPT)
        self.assertIn("--fold 4", SCRIPT)


if __name__ == "__main__":
    unittest.main()
