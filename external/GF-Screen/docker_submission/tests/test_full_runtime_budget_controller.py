from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[4]
SCRIPT = (
    ROOT
    / "external/GF-Screen/docker_submission/run_full_runtime_budget_after_torch113_20260827.sh"
).read_text(encoding="utf-8")


class FullRuntimeBudgetControllerTests(unittest.TestCase):
    def test_waits_for_all_exact_stage_replays_and_is_data_safe(self):
        self.assertIn("biomedparse_torch113_public_equivalence_20260827", SCRIPT)
        self.assertIn("gf_torch113_public_equivalence_20260827", SCRIPT)
        self.assertIn("stunet_torch113_sparse_equivalence_20260827", SCRIPT)
        self.assertIn("summarize_runtime_budget.py", SCRIPT)
        self.assertIn("--hard-seconds 60", SCRIPT)
        self.assertIn("--target-seconds 50", SCRIPT)
        self.assertNotIn("Validation-Public-Labels", SCRIPT)
        self.assertNotIn("Validation-Hidden", SCRIPT)
        self.assertNotIn("package_codabench", SCRIPT)
        self.assertNotIn("codabench.org", SCRIPT)


if __name__ == "__main__":
    unittest.main()
