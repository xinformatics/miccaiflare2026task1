import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "run_focus_encoder_upgrade_routed_tta_20260824.sh").read_text(
    encoding="utf-8"
)


class FocusEncoderUpgradeTTASchedulingTests(unittest.TestCase):
    def test_waits_for_exact_fivefold_upgrade_and_requalifies_tta(self):
        self.assertIn("RESTRICTED_TERMINAL.json", SCRIPT)
        self.assertIn("RESTRICTED_DECISION.json", SCRIPT)
        self.assertIn("PROMOTE_ENCODER_UPGRADE_TO_ROUTED_FULL_VOLUME_OOF", SCRIPT)
        self.assertIn("prepare_focus_encoder_upgrade_tta.py", SCRIPT)
        self.assertIn("evaluate_focus_checkpoint_tta.py", SCRIPT)
        self.assertIn("summarize_focus_routed_tta.py", SCRIPT)

    def test_uses_both_guarded_gpus(self):
        self.assertIn('run_sequence 0 "${specifications0[@]}"', SCRIPT)
        self.assertIn('run_sequence 1 "${specifications1[@]}"', SCRIPT)
        self.assertIn('"${used}" -le 4096', SCRIPT)

    def test_is_local_only(self):
        lowered = SCRIPT.lower()
        self.assertNotIn("codabench", lowered)
        self.assertNotIn("validation-public", lowered)
        self.assertNotIn("validation-hidden", lowered)
        self.assertNotIn(".zip", lowered)
        self.assertIn('"public_read":false', lowered)
        self.assertIn('"hidden_read":false', lowered)


if __name__ == "__main__":
    unittest.main()
