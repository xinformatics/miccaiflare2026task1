#!/usr/bin/env python3
"""Static invariants for the deferred gf_conf top-1 public smoke."""

from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "run_gfconf_top1_refiner_smoke_20260824.sh"


class SmokeSchedulingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_uses_immutable_champion_and_fixed_five_cases(self) -> None:
        self.assertIn("DEPLOYMENT_CHAMPION.json", self.text)
        self.assertIn("cases=(0001 0005 0013 0073 0099)", self.text)
        self.assertIn("--top-k 1", self.text)
        self.assertIn("--expected-count 5", self.text)

    def test_uses_memory_guard_before_inference_without_oof_dependency(self) -> None:
        gpu_wait = self.text.index("nvidia-smi --query-gpu=memory.used")
        inference = self.text.index("predict_seg_from_prompt")
        self.assertLess(gpu_wait, inference)
        self.assertIn('"${used[0]}" -le 16384', self.text)
        self.assertNotIn("recovery_terminal", self.text)

    def test_scores_only_the_exact_smoke_label_subset(self) -> None:
        self.assertIn('"${output}/labels/FLARE23Ts_${case_id}.nii.gz"', self.text)
        self.assertEqual(self.text.count('--gt "${output}/labels"'), 2)
        self.assertNotIn('--gt "${labels}"', self.text)

    def test_has_no_submission_materialization_or_upload_command(self) -> None:
        command_lines = [
            line.strip()
            for line in self.text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        joined = "\n".join(command_lines)
        self.assertNotIn("zip ", joined)
        self.assertNotIn("codabench", joined.lower())
        self.assertNotIn("curl ", joined)
        self.assertNotIn("wget ", joined)


if __name__ == "__main__":
    unittest.main()
