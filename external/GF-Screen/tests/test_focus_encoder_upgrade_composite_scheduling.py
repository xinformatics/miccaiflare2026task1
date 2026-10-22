import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "run_focus_encoder_upgrade_composite_route_decision_20260824.sh"
).read_text(encoding="utf-8")


class FocusEncoderUpgradeCompositeSchedulingTests(unittest.TestCase):
    def test_waits_for_base_upgrade_and_requalified_tta(self):
        self.assertIn("focus_composite_routes_20260824", SCRIPT)
        self.assertIn("RESTRICTED_TERMINAL.json", SCRIPT)
        self.assertIn("RESTRICTED_DECISION.json", SCRIPT)
        self.assertIn("focus_dualwindow_encoder01b8_upgrade_routed_tta", SCRIPT)
        self.assertIn("compose_focus_encoder_upgrade_routes.py", SCRIPT)

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
