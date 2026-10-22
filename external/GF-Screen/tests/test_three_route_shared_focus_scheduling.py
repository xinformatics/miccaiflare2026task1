import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / "docker_submission/run_three_route_shared_focus_public_equivalence_20260827.sh"
).read_text(encoding="utf-8")


class ThreeRouteSharedFocusSchedulingTests(unittest.TestCase):
    def test_serializes_after_sparse_all50_and_biomed(self):
        self.assertIn("biomedparse_dual_prompt_public_equivalence_20260827", SCRIPT)
        self.assertIn("stunet_proxy_sparse_all50_20260827/controller.lock", SCRIPT)
        self.assertIn('exec 8>"${upstream_lock}"', SCRIPT)
        self.assertIn("flock 8", SCRIPT)

    def test_probes_before_all50_and_checks_all_three_routes(self):
        probe = "--part-id 14 --num-parts 50"
        full = "--part-id 0 --num-parts 1"
        self.assertLess(SCRIPT.index(probe), SCRIPT.index(full))
        self.assertIn('(\"guided\",rg/name,og/name)', SCRIPT)
        self.assertIn("exact=not mismatches and not geometry", SCRIPT)

    def test_fail_closed_public_only_contract(self):
        self.assertIn("max(used0,used1)<=1024", SCRIPT)
        self.assertIn("available>=30*1024*1024", SCRIPT)
        self.assertIn("--dataset-role public", SCRIPT)
        self.assertIn('"public_labels_read":False', SCRIPT)
        self.assertIn('"hidden_read":False', SCRIPT)
        for forbidden in ("Validation-Public-Labels", "Validation-Hidden", "requests.post", "curl "):
            self.assertNotIn(forbidden, SCRIPT)


if __name__ == "__main__":
    unittest.main()
