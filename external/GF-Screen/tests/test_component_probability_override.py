from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from materialize_candidate_refiner_component_grid import NumpyOverrideProxy


class ComponentProbabilityOverrideTests(unittest.TestCase):
    def test_only_prediction_classification_is_overridden(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            predictions = work / "predictions"
            patches = work / "patches"
            predictions.mkdir()
            patches.mkdir()
            prediction = predictions / "case__c0001__t000.npz"
            patch = patches / "case__c0001__t000.npz"
            np.savez(
                prediction,
                classification_probability=np.asarray(0.2, dtype=np.float32),
                segmentation_probability=np.ones((1, 2, 2, 2), dtype=np.float16),
            )
            np.savez(patch, classification_probability=np.asarray(0.1))
            proxy = NumpyOverrideProxy(work, {prediction.stem: 0.8})
            with proxy.load(prediction, allow_pickle=False) as archive:
                self.assertAlmostEqual(float(archive["classification_probability"]), 0.8, places=6)
                np.testing.assert_array_equal(
                    archive["segmentation_probability"], np.ones((1, 2, 2, 2), dtype=np.float16)
                )
            with proxy.load(patch, allow_pickle=False) as archive:
                self.assertAlmostEqual(float(archive["classification_probability"]), 0.1)

    def test_unknown_prediction_is_not_overridden(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            predictions = work / "predictions"
            predictions.mkdir()
            path = predictions / "unknown.npz"
            np.savez(path, classification_probability=np.asarray(0.3))
            proxy = NumpyOverrideProxy(work, {})
            with proxy.load(path, allow_pickle=False) as archive:
                self.assertAlmostEqual(float(archive["classification_probability"]), 0.3)


if __name__ == "__main__":
    unittest.main()
