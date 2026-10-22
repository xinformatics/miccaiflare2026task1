import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = (ROOT / "run_source_stratified_e40_e80_public_20260826.sh").read_text()
EVALUATOR = (ROOT / "run_complete_hybrid_e40_e80_public_20260826.sh").read_text()
ARMER = (ROOT / "arm_source_stratified_qualified_hidden_20260826.sh").read_text()
GENERIC_ARMER = (ROOT / "arm_complete_hybrid_qualified_hidden_20260826.sh").read_text()


class SourceStratifiedPublicSchedulingTests(unittest.TestCase):
    def test_public_wrapper_binds_exact_branch_and_model(self):
        self.assertIn('STUNET_EVALUATION_BRANCH="source_stratified_e160"', PUBLIC)
        self.assertIn("STUNetTrainerPanCancerComponentBoxE160SourceStratified", PUBLIC)
        self.assertIn("stunetb_sourcestrat_e40_e80_20260826", PUBLIC)

    def test_evaluator_keeps_fixed_endpoints_and_public_gates(self):
        self.assertIn("epochs=(40 80)", EVALUATOR)
        self.assertIn('keys=("stunet_standalone","gf_anchor_confirm_either_p70")', EVALUATOR)
        self.assertIn("dsc>=0.6366639257651584", EVALUATOR)
        self.assertIn("nsd>=0.5347520263176905", EVALUATOR)
        self.assertIn('branch=="source_stratified_e160"', EVALUATOR)
        self.assertNotIn("Validation-Hidden", EVALUATOR)

    def test_hidden_wrapper_and_armer_fail_closed_on_branch_identity(self):
        self.assertIn('STUNET_EXPECTED_EVALUATION_BRANCH="source_stratified_e160"', ARMER)
        self.assertIn('STUNET_BRANCH_ARCHIVE_PREFIX="gf_ss"', ARMER)
        self.assertIn('endpoint_payload.get("branch")!=expected_branch', GENERIC_ARMER)
        self.assertIn('STUNET_ALLOWED_EPOCHS="40,80"', GENERIC_ARMER)
        self.assertNotIn("codabench.org", ARMER)


if __name__ == "__main__":
    unittest.main()
