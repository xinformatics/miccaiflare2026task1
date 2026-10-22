import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "run_focus_fold4_tta_pilot_20260824.sh").read_text(
    encoding="utf-8"
)


class FocusFold4TTAPilotSchedulingTests(unittest.TestCase):
    def test_is_local_only_and_waits_for_complete_fold(self):
        self.assertIn('${run}/COMPLETE.txt', SCRIPT)
        self.assertIn("audit_focus_fold_epoch_sources.py", SCRIPT)
        self.assertIn("select_focus_single_fold_routes.py", SCRIPT)
        self.assertIn("evaluate_focus_checkpoint_tta.py", SCRIPT)
        self.assertIn("--tta-axes xy", SCRIPT)
        self.assertNotIn("val_image", SCRIPT)
        self.assertNotIn("submission.zip", SCRIPT)
        self.assertNotIn("codabench", SCRIPT.lower())

    def test_uses_guarded_gpu_and_only_selected_epochs(self):
        self.assertIn('used=$(gpu_memory_mib "${gpu}")', SCRIPT)
        self.assertIn('"${used}" -le 4096', SCRIPT)
        self.assertIn('for epoch in "${selected_epochs[@]}"', SCRIPT)
        self.assertIn("weak_sources", SCRIPT)
        self.assertIn("lung_cells", SCRIPT)


if __name__ == "__main__":
    unittest.main()
