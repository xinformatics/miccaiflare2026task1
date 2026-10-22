import unittest
from pathlib import Path


PANCANCER = Path(__file__).resolve().parents[1]
SCRIPT = (PANCANCER / "validate_direct_compact_e100_real_case_20260826.sh").read_text(
    encoding="utf-8"
)


class DirectCompactEquivalenceSchedulingTest(unittest.TestCase):
    def test_waits_for_complete_public_gate_and_is_one_case(self):
        self.assertIn('public_terminal="${legacy_root}/TERMINAL.json"', SCRIPT)
        self.assertLess(SCRIPT.index('[[ -s "${public_terminal}" ]]'), SCRIPT.index('"${python_bin}" -u "${root}/predict_stunet_safe.py"'))
        self.assertIn("--part-id 0 --num-parts 50 --workers 1", SCRIPT.replace("\\\n  ", ""))
        self.assertIn('names[0]!=expected', SCRIPT)

    def test_exercises_exact_opt_in_and_requires_bit_identity(self):
        self.assertIn("--direct-compact-probabilities", SCRIPT)
        self.assertIn('predict.get("direct_compact_probabilities")', SCRIPT)
        self.assertIn("np.array_equal(legacy_mask,direct_mask)", SCRIPT)
        self.assertIn("np.array_equal(legacy_probability,direct_probability)", SCRIPT)
        self.assertIn('"activation_authorized":bool(mask_equal and probability_equal and geometry_equal)', SCRIPT)

    def test_public_only_and_no_submission_side_effect(self):
        self.assertIn("--dataset-role public", SCRIPT)
        self.assertIn('predict.get("hidden_read")', SCRIPT)
        for forbidden in ("Validation-Hidden", "package_codabench", "codabench.org", "curl ", "requests.post"):
            self.assertNotIn(forbidden, SCRIPT)


if __name__ == "__main__":
    unittest.main()
