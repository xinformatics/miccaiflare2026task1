import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT / "docker_submission/run_dual_glance_shared_focus_public_equivalence_20260827.sh"
).read_text(encoding="utf-8")


class SharedFocusEquivalenceSchedulingTests(unittest.TestCase):
    def test_waits_for_sparse_gpu_zero_lock(self):
        self.assertIn('exec 8>"${sparse_lock}"', SCRIPT)
        self.assertIn("flock 8", SCRIPT)
        self.assertIn("nvidia-smi --id=0", SCRIPT)
        self.assertLess(SCRIPT.index("flock 8"), SCRIPT.index("--part-id 14"))

    def test_probe_precedes_all50(self):
        probe = "--part-id 14 --num-parts 50"
        full = "--part-id 0 --num-parts 1"
        mismatch = "shared-Focus one-case mismatch"
        self.assertIn(probe, SCRIPT)
        self.assertIn(mismatch, SCRIPT)
        self.assertIn(full, SCRIPT)
        self.assertLess(SCRIPT.index(probe), SCRIPT.index(mismatch))
        self.assertLess(SCRIPT.index(mismatch), SCRIPT.index(full))

    def test_requires_both_routes_and_all_geometry(self):
        self.assertIn('("bbox",ra/name,oa/name),("cbg",rb/name,ob/name)', SCRIPT)
        self.assertIn("exact=not mismatches and not geometry", SCRIPT)
        self.assertIn("exact_raw_route_equivalence", SCRIPT)

    def test_public_only_and_no_release_side_effect(self):
        self.assertIn("--dataset-role public", SCRIPT)
        self.assertIn('"public_labels_read":False', SCRIPT)
        self.assertIn('"hidden_read":False', SCRIPT)
        self.assertIn('"deployment_authorized":False', SCRIPT)
        for forbidden in ("package_codabench", "requests.post", "codabench.org", "curl "):
            self.assertNotIn(forbidden, SCRIPT)


if __name__ == "__main__":
    unittest.main()
