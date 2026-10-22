import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "run_focus_encoder_upgrade_tta_fold0_epoch6_accelerator_20260824.sh"
).read_text(encoding="utf-8")


class FocusEncoderTTAAcceleratorSchedulingTests(unittest.TestCase):
    def test_writes_exact_final_cache_location_but_not_a_decision(self):
        self.assertIn("predictions/fold0_epoch0006", SCRIPT)
        self.assertIn("evaluate_focus_checkpoint_tta.py", SCRIPT)
        self.assertIn("audit_focus_fold_epoch_sources.py", SCRIPT)
        self.assertNotIn("DECISION.json", SCRIPT)
        self.assertIn("reusable_diagnostic_not_a_promotion_gate", SCRIPT)

    def test_guarded_gpu_and_local_only(self):
        self.assertIn("--id=0", SCRIPT)
        self.assertIn('"${used}" -le 4096', SCRIPT)
        lowered = SCRIPT.lower()
        self.assertNotIn("codabench", lowered)
        self.assertNotIn("validation-public", lowered)
        self.assertNotIn("validation-hidden", lowered)
        self.assertNotIn(".zip", lowered)
        self.assertIn('"public_read":false', lowered)


if __name__ == "__main__":
    unittest.main()
