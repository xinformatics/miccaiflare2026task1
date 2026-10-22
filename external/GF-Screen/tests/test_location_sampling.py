import unittest

import numpy as np

from location_sampling import (
    group_preserving_location_weights,
    location_sampling_audit,
)


class LocationSamplingTests(unittest.TestCase):
    def test_square_root_balance_raises_rare_inferior_quartile(self):
        locations = [0.1] * 5 + [0.4] * 35 + [0.6] * 50 + [0.9] * 10
        groups = ["labeled"] * len(locations)
        audit = location_sampling_audit(locations, groups)
        self.assertAlmostEqual(audit["raw_bin_fraction"][0], 0.05)
        self.assertGreater(audit["balanced_bin_fraction"][0], 0.10)
        self.assertLess(audit["balanced_bin_fraction"][0], 0.15)

    def test_every_source_and_pseudo_group_preserves_exact_mass(self):
        locations = [0.1, 0.3, 0.6, 0.9] * 3
        groups = ["source-a"] * 4 + ["source-b"] * 4 + ["pseudo"] * 4
        base = np.asarray([1, 2, 3, 4] * 3, dtype=float)
        weights = group_preserving_location_weights(locations, groups, base)
        for group in set(groups):
            mask = np.asarray([value == group for value in groups])
            self.assertAlmostEqual(float(weights[mask].sum()), float(base[mask].sum()), places=12)

    def test_zero_power_is_bitwise_unchanged(self):
        locations = [0.1, 0.3, 0.6, 0.9]
        groups = ["a"] * 4
        base = np.asarray([0.5, 1.0, 2.0, 4.0])
        observed = group_preserving_location_weights(
            locations, groups, base, power=0.0
        )
        np.testing.assert_array_equal(observed, base)

    def test_invalid_or_unrepresented_locations_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "represented"):
            group_preserving_location_weights([0.1, 0.2], ["a", "a"])
        with self.assertRaisesRegex(ValueError, "\[0,1\]"):
            group_preserving_location_weights(
                [-0.1, 0.3, 0.6, 0.9], ["a"] * 4
            )


if __name__ == "__main__":
    unittest.main()
