import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "run_source_stunet_precision_pipeline_20260827.sh").read_text()


class SourceStunetPrecisionPipelineTests(unittest.TestCase):
    def test_partial_label_contract_deauthorizes_negative_mining_before_wait(self):
        self.assertIn("LABEL_CONTRACT_CORRECTION_20260825.json", SCRIPT)
        self.assertIn("DEAUTHORIZED_PARTIAL_LABEL_NEGATIVES", SCRIPT)
        self.assertLess(
            SCRIPT.index("DEAUTHORIZED_PARTIAL_LABEL_NEGATIVES"),
            SCRIPT.index('while [[ ! -s "${aggressive}" ]]'),
        )

    def test_runs_all_stages_in_training_to_public_order(self):
        calls = [
            '${proposal_controller}" >>',
            '${training_controller}" >>',
            '${oof_controller}" >>',
            '${public_controller}" >>',
        ]
        positions = [SCRIPT.index(call) for call in calls]
        self.assertEqual(positions, sorted(positions))

    def test_waits_for_prior_aggressive_campaign(self):
        wait = SCRIPT.index('while [[ ! -s "${aggressive}" ]]')
        proposal = SCRIPT.index('${proposal_controller}" >>')
        self.assertLess(wait, proposal)
        self.assertIn('p.get("loss")!={"alpha_fp":0.20,"beta_fn":0.80}', SCRIPT)

    def test_no_upload_and_hidden_is_only_public_qualified(self):
        self.assertIn("run_source_stunet_refiner_public_save_20260827.sh", SCRIPT)
        self.assertIn('"hidden_labels_read":False', SCRIPT)
        self.assertIn('"uploaded":False', SCRIPT)
        for forbidden in ("codabench.org", "requests.post", "curl ", "selenium"):
            self.assertNotIn(forbidden, SCRIPT)

    def test_terminal_is_durable_and_retryable(self):
        self.assertIn("os.replace(t,output)", SCRIPT)
        self.assertIn("RETRY_FROM_VALIDATED_STAGE", SCRIPT)
        self.assertIn("CANDIDATE_SAVED", SCRIPT)
        self.assertIn("KEEP_EPOCH160_P70", SCRIPT)


if __name__ == "__main__":
    unittest.main()
