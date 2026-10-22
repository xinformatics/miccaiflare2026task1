import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from component_confidence_rescue import (  # noqa: E402
    apply_confidence_confirmation,
    configurations,
)


class ComponentConfidenceRescueTests(unittest.TestCase):
    def test_fixed_grid_has_four_distinct_rules(self):
        configs = configurations()
        self.assertEqual(len(configs), 4)
        self.assertEqual(
            {(c["existence_stat"], c["existence_threshold"]) for c in configs},
            {("mean", 0.10), ("mean", 0.30), ("max", 0.10), ("max", 0.30)},
        )

    def test_rescues_only_three_signal_consensus(self):
        base = np.zeros((4, 5, 5), dtype=bool)
        base[0, 0:2, 0:2] = True       # prompt-confirmed
        base[1:3, 3:5, 0:2] = True     # eligible consensus rescue
        base[2:4, 0:2, 3:5] = True     # high Focus but low existence
        prompt = np.zeros_like(base)
        prompt[0, 0, 0] = True
        high = np.zeros_like(base)
        high[1, 3, 0] = True
        high[2, 0, 3] = True
        existence = np.asarray([0.0, 0.4, 0.4, 0.0])
        config = {
            "existence_stat": "mean",
            "existence_threshold": 0.3,
            "rescue_min_mm3": 1,
            "rescue_max_mm3": 100,
        }
        output = apply_confidence_confirmation(
            base, prompt, high, existence, 1.0, config
        )
        self.assertTrue(output[0, 0, 0])
        self.assertTrue(output[1, 3, 0])
        self.assertFalse(output[2, 0, 3])

    def test_rejects_component_outside_size_bound(self):
        base = np.zeros((2, 3, 3), dtype=bool)
        base[:, :2, :2] = True
        config = {
            "existence_stat": "max",
            "existence_threshold": 0.1,
            "rescue_min_mm3": 1,
            "rescue_max_mm3": 4,
        }
        output = apply_confidence_confirmation(
            base, np.zeros_like(base), base, np.ones(2), 1.0, config
        )
        self.assertFalse(output.any())


if __name__ == "__main__":
    unittest.main()
