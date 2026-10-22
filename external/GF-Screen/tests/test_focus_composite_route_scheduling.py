import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "run_focus_composite_route_decision_20260824.sh").read_text(
    encoding="utf-8"
)


class CompositeFocusRouteSchedulingTests(unittest.TestCase):
    def test_waits_for_exact_local_gates_and_composes_all_three_decisions(self):
        self.assertIn("focus_epoch_routes_decision.json", SCRIPT)
        self.assertIn("focus_b32_source_hybrid_decision.json", SCRIPT)
        self.assertIn("focus_dualwindow_cell707_routed_tta_fivefold_20260824", SCRIPT)
        self.assertIn("compose_focus_b32_epoch_tta_routes.py", SCRIPT)
        self.assertIn("QUALIFY_FOCUS_EPOCH_ROUTES_FULL_VOLUME", SCRIPT)
        self.assertIn("KEEP_GLOBAL_FOCUS_EPOCHS", SCRIPT)
        self.assertIn('--epoch-gate "${epoch_gate}"', SCRIPT)
        self.assertIn("QUALIFY_MIXED_FOCUS_B32_FULL_VOLUME", SCRIPT)

    def test_is_local_only_and_does_not_materialize_or_upload(self):
        lowered = SCRIPT.lower()
        self.assertIn('"public_read":false', lowered)
        self.assertIn('"hidden_read":false', lowered)
        self.assertNotIn("codabench", lowered)
        self.assertNotIn("validation-public", lowered)
        self.assertNotIn("validation-hidden", lowered)
        self.assertNotIn(".zip", lowered)


if __name__ == "__main__":
    unittest.main()
