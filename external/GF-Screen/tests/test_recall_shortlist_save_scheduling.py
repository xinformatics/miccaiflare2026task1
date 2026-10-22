import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "save_recall_shortlist_after_public_20260825.sh"
VERIFIER = ROOT / "validate_recall_shortlist_save_contract.py"


class RecallShortlistSaveSchedulingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")
        cls.verifier = VERIFIER.read_text(encoding="utf-8")

    def test_public_contract_and_skip_precede_hidden_transition(self):
        contract = self.text.index("validate_recall_shortlist_save_contract.py")
        skip = self.text.index('if [[ "${contract_status}" == SKIP ]]')
        hidden = self.text.index("Persist the provenance transition")
        self.assertLess(contract, skip)
        self.assertLess(skip, hidden)

    def test_hidden_inference_is_exact_100_and_two_gpu_sharded(self):
        self.assertIn("Validation-Hidden-Images", self.text)
        self.assertIn("expected exact 100 hidden assets", self.text)
        self.assertIn("--num_shards 2 --shard_id", self.text)
        self.assertIn("infer_shard 0 0", self.text)
        self.assertIn("infer_shard 1 1", self.text)

    def test_deployment_ladder_matches_public_screen(self):
        self.assertIn("--glance_threshold 0.20 --lesion_threshold 0.50", self.text)
        self.assertIn("--min-support-fraction 0.25 --min-support-mm3 100", self.text)
        self.assertIn("--configuration touching_component", self.text)
        self.assertIn(
            'minimum = 500 if variant == "deployed500" else 1000',
            self.verifier,
        )

    def test_archive_is_flat_exact_100_and_manual_only(self):
        self.assertIn("--expected-count 100", self.text)
        self.assertIn("len(archive.name)>=64", self.text)
        self.assertIn("archive is not exact flat nonempty NIfTI", self.text)
        self.assertIn("ARCHIVE_READY_FOR_MANUAL_UPLOAD", self.text)
        self.assertNotIn("www.codabench", self.text.lower())
        self.assertNotIn("curl ", self.text)


if __name__ == "__main__":
    unittest.main()
