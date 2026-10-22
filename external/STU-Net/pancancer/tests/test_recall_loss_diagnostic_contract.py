import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "run_source_recall_loss_diagnostic_20260826.sh").read_text()
SELECTOR = (ROOT / "select_recall_loss_escalation.py").read_text()


class RecallLossDiagnosticContractTests(unittest.TestCase):
    def test_waits_for_both_fixed_public_campaigns(self):
        self.assertIn("stunetb_balanced_e40_e80_20260826", SCRIPT)
        self.assertIn("stunetb_sourcestrat_e40_e80_20260826", SCRIPT)
        self.assertIn('p.get("predeclared_epochs")!=[40,80]', SCRIPT)
        self.assertIn("for branch in balanced source", SCRIPT)
        self.assertIn("for epoch in 40 80", SCRIPT)

    def test_uses_voxel_and_component_diagnostics(self):
        self.assertIn("analyze_lesion_errors.py", SCRIPT)
        self.assertIn("analyze_dsc_nsd_gap.py", SCRIPT)
        self.assertIn("select_recall_loss_escalation.py", SCRIPT)
        self.assertIn("FN_ORACLE_ADVANTAGE_MINIMUM = 0.02", SELECTOR)
        self.assertIn("MICRO_RECALL_GAIN_MINIMUM = 0.005", SELECTOR)
        self.assertIn("MICRO_PRECISION_LOSS_MAXIMUM = 0.03", SELECTOR)

    def test_failed_diagnostic_is_retryable_but_complete_is_final(self):
        self.assertIn("COMPLETE) exit 0", SCRIPT)
        self.assertIn("FAILED)", SCRIPT)
        self.assertIn("failed_retry_", SCRIPT)

    def test_has_no_hidden_packaging_or_upload_capability(self):
        for forbidden in (
            "Validation-Hidden",
            "codabench.org",
            "requests.post",
            "package_codabench",
            "curl ",
        ):
            self.assertNotIn(forbidden, SCRIPT)
            self.assertNotIn(forbidden, SELECTOR)
        self.assertIn('"hidden_read": False', SELECTOR)
        self.assertIn('"submission_created": False', SELECTOR)
        self.assertIn('"uploaded": False', SELECTOR)


if __name__ == "__main__":
    unittest.main()
