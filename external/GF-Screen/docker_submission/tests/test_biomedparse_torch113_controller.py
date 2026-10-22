from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = (
    ROOT
    / "external/GF-Screen/docker_submission/run_biomedparse_torch113_public_equivalence_20260827.sh"
).read_text(encoding="utf-8")


class Torch113ControllerTest(unittest.TestCase):
    def test_is_serialized_and_label_free(self):
        self.assertIn("runtime_anchor_equivalence_20260827", SCRIPT)
        self.assertIn("flock 8", SCRIPT)
        self.assertIn("--network none", SCRIPT)
        self.assertIn("${project}:/project:ro", SCRIPT)
        self.assertNotIn("Validation-Public-Labels", SCRIPT)
        self.assertNotIn("Validation-Hidden", SCRIPT)
        self.assertNotIn("package_codabench", SCRIPT)
        self.assertNotIn("codabench.org", SCRIPT)
        self.assertNotIn("requests.post", SCRIPT)

    def test_requires_exact_probe_before_all50(self):
        probe_check = SCRIPT.index("torch-1.13 one-case mismatch")
        full_run = SCRIPT.index('run_container "${full}"')
        self.assertLess(probe_check, full_run)
        self.assertIn('p.get("torch_version")!="1.13.1+cu116"', SCRIPT)
        self.assertIn("np.array_equal", SCRIPT)
        self.assertIn("exact_prompt_outputs", SCRIPT)
        self.assertIn("peak_gpu_memory_mb", SCRIPT)
        self.assertIn("memory_eligible_16gb", SCRIPT)
        self.assertIn("memory_eligible_4gb_efficiency", SCRIPT)


if __name__ == "__main__":
    unittest.main()
