import unittest

from nnunet.summarize_training_partial_region_strata import summarize


class TrainingPartialRegionStrataTest(unittest.TestCase):
    def test_summary_is_deterministic_and_reports_low_ratio_fraction(self):
        result = summarize([1.0, 3.0, 5.0, 7.0, 9.0])
        self.assertEqual(result["patches"], 5)
        self.assertEqual(
            result["background_to_foreground_ratio_quantiles"]["0.5"], 5.0
        )
        self.assertEqual(result["fraction_below_5"], 0.4)

    def test_empty_stratum_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty stratum"):
            summarize([])


if __name__ == "__main__":
    unittest.main()
