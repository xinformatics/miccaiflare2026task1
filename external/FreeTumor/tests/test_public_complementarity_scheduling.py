import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = PROJECT_ROOT / "external" / "FreeTumor" / "run_public_complementarity_after_pipeline_20260825.sh"


class PublicComplementaritySchedulingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_waits_behind_checkpoint_303_submission_screen(self):
        self.assertIn(
            "gf_focus_pu_e303_interim_saved_20260825/TERMINAL.json",
            self.text,
        )
        self.assertIn("COMPLETE|SKIPPED", self.text)

    def test_binds_published_checkpoint_identity(self):
        self.assertIn(
            "c039619094c8b9b38a06bd62d8b5c8f088507cfa39419120ec8beaba255c05a0",
            self.text,
        )
        self.assertIn("model_baseline_segmentor.pt", self.text)

    def test_uses_both_gpus_and_exact_public_inventory(self):
        self.assertIn("for shard in 0 1", self.text)
        self.assertIn("--num-shards 2", self.text)
        self.assertIn("len(image_names)!=50", self.text)
        self.assertIn("Validation-Public-Images", self.text)
        self.assertIn("Validation-Public-Labels", self.text)

    def test_submission_and_hidden_operations_are_unreachable(self):
        lowered = self.text.lower()
        self.assertNotIn("validation-hidden", lowered)
        self.assertNotIn("codabench", lowered)
        self.assertNotIn("zip -", lowered)
        self.assertNotIn("curl ", lowered)
        self.assertNotIn("wget ", lowered)
        self.assertIn('"submission_created":False', self.text)
        self.assertIn('"uploaded":False', self.text)


if __name__ == "__main__":
    unittest.main()
