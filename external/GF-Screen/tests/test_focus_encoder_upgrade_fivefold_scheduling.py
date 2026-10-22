import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "run_focus_dualwindow_encoder_upgrade_fivefold_20260824.sh"
).read_text(encoding="utf-8")


class FocusEncoderUpgradeFivefoldSchedulingTests(unittest.TestCase):
    def test_waits_for_route_aware_fold0_gate(self):
        self.assertIn("ROUTE_TERMINAL.json", SCRIPT)
        self.assertIn("PROMOTE_ROUTES_TO_FIVEFOLD", SCRIPT)

    def test_splits_untouched_folds_over_both_guarded_gpus(self):
        self.assertIn('train_sequence "${gpu0}" 1 3', SCRIPT)
        self.assertIn('train_sequence "${gpu1}" 2 4', SCRIPT)
        self.assertIn('"${used}" -le 4096', SCRIPT)
        self.assertIn('wait "${pid0}"', SCRIPT)
        self.assertIn('wait "${pid1}"', SCRIPT)

    def test_is_training_only(self):
        lowered = SCRIPT.lower()
        self.assertNotIn("codabench", lowered)
        self.assertNotIn("validation-public", lowered)
        self.assertNotIn("validation-hidden", lowered)
        self.assertNotIn(".zip", lowered)


if __name__ == "__main__":
    unittest.main()
