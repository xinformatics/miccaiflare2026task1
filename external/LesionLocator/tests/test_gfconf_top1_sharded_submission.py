#!/usr/bin/env python3
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "run_gfconf_top1_submission_sharded_20260824.sh"


class ShardedSubmissionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_requires_fixed_smoke_and_prompt_audits(self) -> None:
        self.assertIn("QUALIFY_FOR_FULL_PUBLIC", self.text)
        self.assertIn('p["cases"]!=50', self.text)
        self.assertIn('h["cases"]!=100', self.text)

    def test_runs_balanced_two_gpu_shards(self) -> None:
        self.assertIn("for gpu in 0 1", self.text)
        self.assertIn('CUDA_VISIBLE_DEVICES="${gpu}"', self.text)
        self.assertIn("expected//2", self.text)

    def test_public_gate_precedes_hidden_and_package(self) -> None:
        public = self.text.index("public_seconds=$(run_pair public)")
        gate = self.text.index("public_gate=")
        hidden = self.text.index("hidden_seconds=$(run_pair hidden)")
        package = self.text.index("package_codabench.py")
        self.assertLess(public, gate)
        self.assertLess(gate, hidden)
        self.assertLess(hidden, package)

    def test_recipe_matches_public_and_hidden(self) -> None:
        for option in (
            "--min-seed-coverage 0.25", "--min-refined-overlap 0.25",
            "--min-volume-ratio 0.25", "--max-volume-ratio 4.0",
        ):
            self.assertEqual(self.text.count(option), 2)

    def test_no_upload_command_or_champion_mutation(self) -> None:
        self.assertNotIn("curl ", self.text)
        self.assertNotIn("wget ", self.text)
        self.assertNotIn('> "${champion}"', self.text)


if __name__ == "__main__":
    unittest.main()
