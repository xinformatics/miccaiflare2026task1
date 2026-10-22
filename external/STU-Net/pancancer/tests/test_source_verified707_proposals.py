import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "run_source_stunet_verified707_proposals_20260827.sh").read_text()


class SourceVerified707ProposalTests(unittest.TestCase):
    def test_uses_selected_source_endpoint_and_exact_complete_manifest(self):
        self.assertIn("stunetb_sourcestrat_e40_e80_20260826", SCRIPT)
        self.assertIn("fpr_training_mining_707_20260822/manifest.json", SCRIPT)
        self.assertIn('len(items)!=707', SCRIPT)
        self.assertIn('item.get("complete_label") is not True', SCRIPT)
        self.assertIn("checkpoint_inference_epoch_", SCRIPT)

    def test_is_two_gpu_mask_only_training_inference(self):
        self.assertIn("run_shard 0 0 &", SCRIPT)
        self.assertIn("run_shard 1 1 &", SCRIPT)
        self.assertIn("--dataset-role training_complete", SCRIPT)
        self.assertNotIn("--save-probabilities", SCRIPT)
        self.assertIn('"probabilities_saved":False', SCRIPT)
        self.assertIn('"training_labels_read":False', SCRIPT)
        self.assertIn("FAILED)", SCRIPT)
        self.assertIn("failed_retry_", SCRIPT)

    def test_cannot_read_validation_or_package_or_upload(self):
        for forbidden in (
            "Validation-Public",
            "Validation-Hidden",
            "package_codabench",
            "codabench.org",
            "requests.post",
            "curl ",
        ):
            self.assertNotIn(forbidden, SCRIPT)
        self.assertIn('p.get("public_labels_read")', SCRIPT)
        self.assertIn('p.get("hidden_labels_read")', SCRIPT)
        self.assertIn('p.get("uploaded")', SCRIPT)


if __name__ == "__main__":
    unittest.main()
