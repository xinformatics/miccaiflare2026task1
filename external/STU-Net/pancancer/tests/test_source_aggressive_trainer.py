import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAINER = (ROOT / "STUNetTrainerPanCancerSourceAggressive.py").read_text()
RUNNER = (ROOT / "run_pancancer_stunet_source_aggressive.py").read_text()


class SourceAggressiveTrainerTests(unittest.TestCase):
    def test_changes_only_loss_on_audited_source_trainer(self):
        self.assertIn(
            "STUNetTrainerPanCancerComponentBoxE160SourceStratified\n)", TRAINER
        )
        self.assertIn("PartialBoxTverskyCELoss(alpha=0.20, beta=0.80)", TRAINER)
        self.assertNotIn("alpha=0.25, beta=0.75", TRAINER)
        self.assertNotIn("def get_dataloaders", TRAINER)
        self.assertNotIn("def __init__", TRAINER)

    def test_keeps_distinct_tracking_and_gate_identity(self):
        self.assertIn("stunetbcompe160sourcea020b08020260826", TRAINER)
        self.assertIn("QUALIFY_TVERSKY_A020_B080_FOLLOWUP", TRAINER)
        self.assertIn('"parent_epoch": 160', TRAINER)
        self.assertIn('"optimizer_restart": True', TRAINER)

    def test_runner_resolves_only_the_new_class(self):
        name = "STUNetTrainerPanCancerComponentBoxE160SourceAggressive"
        self.assertIn(f'TRAINER_NAME = "{name}"', RUNNER)
        self.assertIn("load_stunet_pretrained_weights", RUNNER)
        self.assertIn("run_training_entry()", RUNNER)


if __name__ == "__main__":
    unittest.main()
