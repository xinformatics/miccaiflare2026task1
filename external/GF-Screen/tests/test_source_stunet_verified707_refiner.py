import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (ROOT / "run_source_stunet_verified707_refiner_20260827.sh").read_text()
SELECTOR = (ROOT / "select_stunet_proposal_refiner_oof.py").read_text()


class SourceStunetVerified707RefinerTests(unittest.TestCase):
    def test_direct_launch_is_deauthorized_before_patch_or_gpu_work(self):
        self.assertIn("LABEL_CONTRACT_CORRECTION_20260825.json", SCRIPT)
        self.assertIn("DEAUTHORIZED_PARTIAL_LABEL_NEGATIVES", SCRIPT)
        self.assertLess(
            SCRIPT.index("DEAUTHORIZED_PARTIAL_LABEL_NEGATIVES"),
            SCRIPT.index('while [[ ! -s "${proposal_terminal}" ]]'),
        )

    def test_training_uses_exact_complete_cohort_and_stu_only_source(self):
        self.assertIn("fpr_training_mining_707_20260822/manifest.json", SCRIPT)
        self.assertIn("source_stunet_verified707_proposals_20260827", SCRIPT)
        self.assertIn('--proposal "source_stunet=${proposals}"', SCRIPT)
        self.assertIn('summary.get("cases")==707', SCRIPT)
        self.assertIn('{row["source_name"] for row in rows}=={"source_stunet"}', SCRIPT)

    def test_recipe_is_fivefold_and_reproducible(self):
        self.assertIn("--folds 5", SCRIPT)
        self.assertIn("--epochs 30", SCRIPT)
        self.assertIn("--batch-size 8", SCRIPT)
        self.assertIn("--lr 3e-4", SCRIPT)
        self.assertIn("--base-channels 32", SCRIPT)
        self.assertIn("--samples-per-epoch 8192", SCRIPT)
        self.assertIn("--seed 20260827", SCRIPT)
        self.assertIn("--resume", SCRIPT)

    def test_deployment_is_classifier_selected_seed_contour(self):
        self.assertIn("checkpoint_best_classification.pt", SCRIPT)
        self.assertIn("validation_predictions_best_classification.npz", SCRIPT)
        self.assertIn('"deployment_role":"classify source-STU components/tiles; preserve accepted STU seed contour"', SCRIPT)
        self.assertIn('"deployment_mode": "seed"', SELECTOR)

    def test_precision_gate_is_strict_and_public_blind(self):
        self.assertIn("PRECISION_MINIMUM = 0.90", SELECTOR)
        self.assertIn("PRECISION_LOWER_CI_MINIMUM = 0.86", SELECTOR)
        self.assertIn("RECALL_MINIMUM = 0.15", SELECTOR)
        self.assertIn("AP_LIFT_MINIMUM = 0.20", SELECTOR)
        self.assertIn("QUALIFY_STUNET_SEED_FULL_VOLUME_OOF", SELECTOR)
        for forbidden in ("Validation-Public", "Validation-Hidden", "codabench.org", "curl "):
            self.assertNotIn(forbidden, SCRIPT)
            self.assertNotIn(forbidden, SELECTOR)

    def test_no_upload_or_submission_creation(self):
        self.assertIn('"public_read":False', SCRIPT)
        self.assertIn('"hidden_read":False', SCRIPT)
        self.assertIn('"submission_created":False', SCRIPT)
        self.assertIn('"uploaded":False', SCRIPT)


if __name__ == "__main__":
    unittest.main()
