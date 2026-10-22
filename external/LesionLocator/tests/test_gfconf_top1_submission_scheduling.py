#!/usr/bin/env python3
"""Static safety and ordering checks for the top-1 submission controller."""

from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "run_gfconf_top1_submission_20260824.sh"


class SubmissionSchedulingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_uses_champion_and_exact_inventories(self) -> None:
        self.assertIn("DEPLOYMENT_CHAMPION.json", self.text)
        self.assertIn("--expected-count 50", self.text)
        self.assertIn("--expected-count 100", self.text)
        self.assertIn("--top-k 1", self.text)

    def test_public_gate_precedes_hidden_inference(self) -> None:
        smoke = self.text.index('smoke_decision=')
        public_inference = self.text.index('public_start=$(date +%s)')
        public_gate = self.text.index('public_gate=')
        hidden_inference = self.text.index('hidden_start=$(date +%s)')
        package = self.text.index('package_codabench.py')
        self.assertLess(smoke, public_inference)
        self.assertLess(public_inference, public_gate)
        self.assertLess(public_gate, hidden_inference)
        self.assertLess(hidden_inference, package)

    def test_fixed_recipe_is_identical_for_public_and_hidden(self) -> None:
        self.assertEqual(self.text.count("--min-seed-coverage 0.25"), 2)
        self.assertEqual(self.text.count("--min-refined-overlap 0.25"), 2)
        self.assertEqual(self.text.count("--min-volume-ratio 0.25"), 2)
        self.assertEqual(self.text.count("--max-volume-ratio 4.0"), 2)

    def test_never_uploads_or_mutates_champion(self) -> None:
        command_lines = [
            line.strip() for line in self.text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        joined = "\n".join(command_lines)
        self.assertNotIn("curl ", joined)
        self.assertNotIn("wget ", joined)
        self.assertNotIn("requests.post", joined)
        self.assertNotIn("playwright", joined.lower())
        self.assertNotIn('> "${champion}"', joined)


if __name__ == "__main__":
    unittest.main()
