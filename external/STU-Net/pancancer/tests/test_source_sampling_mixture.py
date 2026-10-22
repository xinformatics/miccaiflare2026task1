import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "analyze_source_sampling_mixture.py"
SPEC = importlib.util.spec_from_file_location("source_sampling_mixture", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SourceSamplingMixtureTests(unittest.TestCase):
    def test_source_parser_preserves_embedded_underscores(self):
        self.assertEqual(
            MODULE.source_from_identifier("WholeBody_LongitudinalCTLesion_abc-FU-00"),
            "WholeBody_LongitudinalCTLesion",
        )

    def test_half_mixture_is_normalized_and_bounded(self):
        identifiers = ["large_0", "large_1", "large_2", "small_0"]
        audit = MODULE.build_audit(identifiers, source_fraction=0.5, draws_per_epoch=8)
        probabilities = np.asarray(
            [row["sampling_probability"] for row in audit["identifiers"]]
        )
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        self.assertGreater(probabilities[-1], probabilities[0])
        self.assertLess(probabilities[-1] / probabilities[0], 3.0)
        by_source = {row["source"]: row for row in audit["sources"]}
        self.assertAlmostEqual(
            by_source["small"]["proposed_expected_draws_per_epoch"], 3.0
        )
        self.assertAlmostEqual(
            by_source["large"]["proposed_expected_draws_per_epoch"], 5.0
        )

    def test_endpoints_match_uniform_case_and_uniform_source(self):
        identifiers = ["large_0", "large_1", "large_2", "small_0"]
        uniform_case = MODULE.build_audit(identifiers, 0.0, 4)
        uniform_source = MODULE.build_audit(identifiers, 1.0, 4)
        self.assertTrue(
            np.allclose(
                [row["sampling_probability"] for row in uniform_case["identifiers"]],
                [0.25] * 4,
            )
        )
        by_source = {row["source"]: row for row in uniform_source["sources"]}
        self.assertAlmostEqual(by_source["large"]["proposed_probability_mass"], 0.5)
        self.assertAlmostEqual(by_source["small"]["proposed_probability_mass"], 0.5)

    def test_invalid_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            MODULE.build_audit([], 0.5, 500)
        with self.assertRaises(ValueError):
            MODULE.build_audit(["bad"], 0.5, 500)
        with self.assertRaises(ValueError):
            MODULE.build_audit(["source_0"], 1.1, 500)


if __name__ == "__main__":
    unittest.main()
