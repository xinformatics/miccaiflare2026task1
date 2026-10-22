import unittest
from pathlib import Path


PANCANCER = Path(__file__).resolve().parents[1]
SCRIPT = (
    PANCANCER / "run_anchor_sparse_e160_public_equivalence_20260827.sh"
).read_text(encoding="utf-8")


class AnchorSparseAll50SchedulingTests(unittest.TestCase):
    def test_requires_qualified_probe_before_any_predictor(self):
        validation = 'if (p.get("status")!="COMPLETE" or not p.get("probe_pass")'
        predictor = 'predict_stunet_anchor_sparse.py'
        self.assertIn(validation, SCRIPT)
        self.assertLess(SCRIPT.index(validation), SCRIPT.index(predictor))
        self.assertIn('not p.get("runtime_eligible_55s")', SCRIPT)

    def test_exact_50_sequential_case_contract(self):
        self.assertIn('for part_id in $(seq 0 49)', SCRIPT)
        self.assertIn('--num-parts 50', SCRIPT)
        self.assertIn('case_count")!=1', SCRIPT)
        self.assertIn("if len(names)!=50", SCRIPT)

    def test_authorization_needs_all_mask_geometry_and_runtime_gates(self):
        self.assertIn('exact=not mismatches and not geometry_failures', SCRIPT)
        self.assertIn('runtime=max(times)<=55.0', SCRIPT)
        self.assertIn('authorized=bool(exact and runtime)', SCRIPT)
        self.assertIn('AUTHORIZE_SPARSE_DEPLOYMENT', SCRIPT)

    def test_public_only_and_no_submission_side_effect(self):
        self.assertIn('--dataset-role public', SCRIPT)
        self.assertIn('"public_labels_read":False', SCRIPT)
        self.assertIn('"hidden_read":False', SCRIPT)
        for forbidden in (
            "Validation-Hidden",
            "package_codabench",
            "codabench.org",
            "requests.post",
            "curl ",
        ):
            self.assertNotIn(forbidden, SCRIPT)


if __name__ == "__main__":
    unittest.main()
