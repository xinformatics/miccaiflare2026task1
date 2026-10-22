import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAVER = (ROOT / "save_candidate_refiner_v3_weak707_candidate_20260823.sh").read_text()
WATCHER = (ROOT / "save_focus_encoder_notta_after_public_20260825.sh").read_text()


class FocusEncoderNoTtaSaveSchedulingTest(unittest.TestCase):
    def test_generic_saver_reproduces_composite_route(self):
        self.assertIn("WEAK707_SAVE_COMPOSITE_ROUTE_DECISION", SAVER)
        self.assertIn("focus_composite_route_selection.py", SAVER)
        self.assertIn('--candidate-contour-route "b32=${checkpoint}"', SAVER)
        self.assertIn('--focus-contour-route "${name}=${value}"', SAVER)
        self.assertIn('--source-contour-route "${name}=${value}"', SAVER)
        self.assertIn('composite_route_decision=${composite_route_decision}', SAVER)

    def test_watcher_requires_material_public_gate(self):
        self.assertIn('dsc>=0.6154316379500749', WATCHER)
        self.assertIn('nsd>=0.5104225107769019', WATCHER)
        self.assertIn('gate.get("decision")=="PROMOTE"', WATCHER)
        self.assertIn('("fixed_public_anchor","paired_gate")', WATCHER)
        self.assertIn("QUALIFY_SUBMISSION_PREPARATION", WATCHER)
        self.assertIn("KEEP_TOUCHCOMP", WATCHER)
        self.assertIn("public decision/gate mismatch", WATCHER)
        self.assertIn('raise SystemExit(f"{key} identity mismatch")', WATCHER)

    def test_watcher_is_save_only_and_exact_100(self):
        self.assertIn("archive coverage mismatch", WATCHER)
        self.assertIn("members\":100", WATCHER)
        self.assertIn('"uploaded":False', WATCHER)
        self.assertNotIn("codabench.org", WATCHER.lower())
        self.assertNotIn("curl ", WATCHER.lower())
        self.assertNotIn("requests.", WATCHER.lower())


if __name__ == "__main__":
    unittest.main()
