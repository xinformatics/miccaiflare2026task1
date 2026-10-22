import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "run_internal_grid_after_pipeline_20260825.sh"


class InternalGridAfterPipelineSchedulingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_waits_for_last_gpu_consumer(self):
        self.assertIn(
            "gf_glance_grl_long30_saved_20260825/TERMINAL.json", self.text
        )
        wait = self.text.index("waiting for final Glance saver terminal")
        launch = self.text.index("evaluate_internal_checkpoint_grid.py")
        self.assertLess(wait, launch)

    def test_requires_both_gpus_to_be_idle(self):
        self.assertIn("nvidia-smi --id=0", self.text)
        self.assertIn("nvidia-smi --id=1", self.text)
        self.assertIn('"${used0}" -le "${idle_memory_mib}"', self.text)
        self.assertIn('"${used1}" -le "${idle_memory_mib}"', self.text)

    def test_complete_range_and_two_gpu_contract_are_explicit(self):
        self.assertIn("--start-epoch 284 --end-epoch 313", self.text)
        self.assertIn("--gpus 0 1", self.text)
        self.assertIn('payload.get("evaluated_checkpoints")!=30', self.text)
        self.assertIn(
            'for key in ("learning_curve", "plot", "fixed_public_shortlist")',
            self.text,
        )
        self.assertIn('asset=payload.get(key,{})', self.text)
        self.assertIn(
            'curve.get("recall_available_checkpoint_count")!=24', self.text
        )
        self.assertIn(
            '"best_annotated_large_component_recall_ge3000"', self.text
        )
        self.assertIn('"best_annotated_voxel_recall"', self.text)
        self.assertIn(
            "case_balanced_dice_plus_ge3000mm3_and_voxel_recall", self.text
        )
        self.assertIn('large_component_minimum_mm3")!=3000', self.text)
        self.assertIn(
            "best_large_component_recall_ge3000_within_dice_guard", self.text
        )
        self.assertIn('"promotion_authority")!="fixed_public_50_only"', self.text)
        self.assertIn("--parity-epoch 295 --parity-tolerance 1e-6", self.text)
        self.assertIn('parity.get("status")!="PASS"', self.text)

    def test_cannot_use_challenge_or_submission_outputs(self):
        self.assertNotIn("Validation-Public", self.text)
        self.assertNotIn("Validation-Hidden", self.text)
        self.assertNotIn("zip ", self.text)
        self.assertNotIn("codabench", self.text.lower())
        self.assertIn('payload.get("public_read")', self.text)
        self.assertIn('payload.get("hidden_read")', self.text)

    def test_failure_is_terminal_and_retryable(self):
        self.assertIn('"status":"FAILED"', self.text)
        self.assertIn("TERMINAL.unverified_", self.text)
        self.assertIn("TERMINAL.failed_retry_", self.text)
        self.assertIn("trap - EXIT", self.text)


if __name__ == "__main__":
    unittest.main()
