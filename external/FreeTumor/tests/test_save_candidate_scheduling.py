import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = PROJECT_ROOT / "external" / "FreeTumor" / "save_public_qualified_candidate_20260825.sh"


class SaveCandidateSchedulingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_public_gate_precedes_hidden_transition(self):
        self.assertLess(
            self.text.index('if [[ "${gate}" != PASS ]]'),
            self.text.index('"${hidden_started}" <<\'PY\''),
        )
        self.assertIn("0.6154316379500749", self.text)
        self.assertIn("0.5104225107769019", self.text)
        self.assertIn("direct_union_public_gate", self.text)

    def test_exact_public_configuration_is_replayed(self):
        self.assertIn('union_mm(0|500|1000|3000)', self.text)
        self.assertIn('--min-mm3 "${minimum}"', self.text)
        self.assertIn("--fusion union", self.text)
        self.assertIn("gf_touchcomp_0825.zip", self.text)

    def test_two_gpu_hidden_and_exact_archive(self):
        self.assertIn("for shard in 0 1", self.text)
        self.assertIn("--dataset-role hidden", self.text)
        self.assertIn("--expected-count 100", self.text)
        self.assertIn("len(members)!=100", self.text)
        self.assertIn('"uploaded":False', self.text)

    def test_no_upload_or_hidden_labels(self):
        for forbidden in ("codabench.org", "requests.post", "curl ", "selenium", "Validation-Hidden-Labels"):
            self.assertNotIn(forbidden, self.text)


if __name__ == "__main__":
    unittest.main()
