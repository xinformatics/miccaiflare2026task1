import re
import unittest
from pathlib import Path

from pseudo_exposure import exposure_fraction, select_integer_repeat


class PseudoExposureTests(unittest.TestCase):
    def test_avoids_upward_rounding_discontinuity(self):
        # The continuous optimum is just above two. Ceil would select three
        # and expose 27.3% pseudo cases; nearest-integer selection stays at 20%.
        repeat = select_integer_repeat(2180, 17445, 0.20)
        self.assertEqual(repeat, 2)
        self.assertAlmostEqual(exposure_fraction(2180, 17445, repeat), 0.20, places=3)

    def test_selects_three_when_three_is_nearest(self):
        self.assertEqual(select_integer_repeat(1454, 17445, 0.20), 3)

    def test_repeat_bounds_are_respected(self):
        self.assertEqual(select_integer_repeat(100, 17445, 0.20), 8)
        self.assertEqual(select_integer_repeat(20000, 17445, 0.20), 1)

    def test_invalid_inputs_fail_closed(self):
        for arguments in (
            (0, 17445, 0.20),
            (100, 0, 0.20),
            (100, 17445, 0.0),
            (100, 17445, 1.0),
        ):
            with self.assertRaises(ValueError):
                select_integer_repeat(*arguments)
        with self.assertRaises(ValueError):
            select_integer_repeat(100, 17445, 0.20, minimum=3, maximum=2)

    def test_long_controller_heredocs_compile_and_use_shared_selector(self):
        root = Path(__file__).resolve().parents[1]
        for filename in (
            "run_long_focus_pu_after_direct_public_20260825.sh",
            "run_long_glance_grl_after_focus_20260825.sh",
        ):
            source = (root / filename).read_text(encoding="utf-8")
            self.assertIn("from pseudo_exposure import", source)
            blocks = re.findall(r"<<'PY'\n(.*?)\nPY\n", source, flags=re.S)
            self.assertTrue(blocks)
            for index, block in enumerate(blocks):
                compile(block, f"{filename}-heredoc-{index}", "exec")


if __name__ == "__main__":
    unittest.main()
