import unittest

from audit_hysteresis_teacher_snapshot import summarize_candidate_mass


class HysteresisTeacherSnapshotTests(unittest.TestCase):
    def test_exact_uniform_preserves_component_fraction(self):
        summary = summarize_candidate_mass([100.0, 400.0, 4000.0], 16, 0.0)
        self.assertEqual(summary["candidate_centers"], 48)
        self.assertEqual(summary["candidate_center_volume_bins"]["100_to_500"], 32)
        self.assertAlmostEqual(summary["small_candidate_fraction_lt_500mm3"], 2 / 3)

    def test_square_root_retains_bounded_small_component_mass(self):
        summary = summarize_candidate_mass([100.0, 10_000.0], 64, 0.5)
        self.assertEqual(summary["candidate_centers"], 128)
        small = summary["candidate_center_volume_bins"]["100_to_500"]
        self.assertGreater(small, 1)
        self.assertLess(small, 64)
        self.assertAlmostEqual(summary["small_candidate_fraction_lt_500mm3"], small / 128)

    def test_empty_component_set_is_defined(self):
        summary = summarize_candidate_mass([], 64, 0.5)
        self.assertEqual(summary["candidate_centers"], 0)
        self.assertEqual(summary["small_candidate_fraction_lt_500mm3"], 0.0)


if __name__ == "__main__":
    unittest.main()
