from pathlib import Path
import unittest


SCRIPT = (Path(__file__).resolve().parents[1] / "run_stunet_torch113_sparse_equivalence_20260827.sh").read_text(encoding="utf-8")


class STUNetTorch113ControllerTests(unittest.TestCase):
    def test_is_serialized_label_free_and_probe_gated(self):
        self.assertIn("gf_torch113_public_equivalence_20260827", SCRIPT)
        self.assertIn("flock 8", SCRIPT)
        self.assertIn("--network none", SCRIPT)
        self.assertIn("${project}:/project:ro", SCRIPT)
        self.assertIn("runtime_eligible_20s", SCRIPT)
        self.assertIn("peak_gpu_memory_mb", SCRIPT)
        self.assertIn("memory_eligible_16gb", SCRIPT)
        self.assertIn("memory_eligible_4gb_efficiency", SCRIPT)
        self.assertLess(SCRIPT.index('run_case "${probe}"'), SCRIPT.index("for part in $(seq 0 49)"))
        self.assertNotIn("Validation-Public-Labels", SCRIPT)
        self.assertNotIn("Validation-Hidden", SCRIPT)
        self.assertNotIn("codabench.org", SCRIPT)
        self.assertNotIn("package_codabench", SCRIPT)


if __name__ == "__main__":
    unittest.main()
