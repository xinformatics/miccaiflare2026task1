import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1] / "run_balanced_e40_tail_gpu0_20260826.sh"
).read_text(encoding="utf-8")


class BalancedE40TailGPU0Tests(unittest.TestCase):
    def test_helper_is_bound_to_disjoint_tail_and_exact_checkpoint(self):
        self.assertIn('part_start="${BALANCED_TAIL_PART_START:-40}"', SCRIPT)
        self.assertIn('part_end="${BALANCED_TAIL_PART_END:-49}"', SCRIPT)
        self.assertIn("part_start >= 35", SCRIPT)
        self.assertIn("part_end <= 49", SCRIPT)
        self.assertIn('for part_id in $(seq "${part_start}" "${part_end}")', SCRIPT)
        self.assertIn("--part-id \"${part_id}\" --num-parts 50", SCRIPT)
        self.assertIn(
            "3015cdf2d9b267989fd2fde2939cdac8670c55c31b96fd50f59eadf9da2a29f6",
            SCRIPT,
        )

    def test_outputs_are_isolated_and_terminal_is_published_last(self):
        self.assertIn('--output "${scratch}/predictions"', SCRIPT)
        mask_move = SCRIPT.index('mv "${mask}" "${predictions}/${names[0]}"')
        probability_move = SCRIPT.index(
            'mv "${probability}" "${predictions}/${names[1]}"'
        )
        terminal_move = SCRIPT.index('mv "${scratch_terminal}" "${target_terminal}"')
        self.assertLess(mask_move, terminal_move)
        self.assertLess(probability_move, terminal_move)

    def test_no_labels_hidden_or_upload_path(self):
        for forbidden in (
            "Validation-Public-Labels",
            "Validation-Hidden",
            "codabench.org",
            "requests.post",
            "curl ",
        ):
            self.assertNotIn(forbidden, SCRIPT)


if __name__ == "__main__":
    unittest.main()
