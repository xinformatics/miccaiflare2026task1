import sys
import unittest
from collections import Counter
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from select_healthy_proxy import cohort, proportional_quotas, select_paths


class HealthyProxySelectionTests(unittest.TestCase):
    def setUp(self):
        self.paths = (
            [Path(f"CHAOS_{index:02d}_0000.nii.gz") for index in range(40)]
            + [Path(f"train_{index}_a_1.nii.gz") for index in range(50)]
            + [Path(f"PANCREAS_{index:04d}_0000.nii.gz") for index in range(82)]
        )

    def test_proportional_fixed_size(self):
        selected, quotas = select_paths(self.paths, total=32, seed="fixed")
        self.assertEqual(len(selected), 32)
        self.assertEqual(quotas, {"CHAOS": 8, "train": 9, "PANCREAS": 15})
        self.assertEqual(Counter(cohort(path) for path in selected), Counter(quotas))

    def test_selection_is_order_independent(self):
        forward, _ = select_paths(self.paths, total=32, seed="fixed")
        reverse, _ = select_paths(list(reversed(self.paths)), total=32, seed="fixed")
        self.assertEqual(forward, reverse)

    def test_seed_changes_selection(self):
        first, _ = select_paths(self.paths, total=32, seed="one")
        second, _ = select_paths(self.paths, total=32, seed="two")
        self.assertNotEqual(first, second)

    def test_invalid_total(self):
        groups = {"A": [Path("A_1.nii.gz")]}
        with self.assertRaises(ValueError):
            proportional_quotas(groups, 2)


if __name__ == "__main__":
    unittest.main()
