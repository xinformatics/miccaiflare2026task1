import unittest

import numpy as np

from compose_candidate_refiner_validation_predictions import compose


def archive(probabilities=(0.2, 0.8), dice=0.4):
    return {
        "patch_id": np.asarray(["a", "b"]),
        "label": np.asarray([0, 1], dtype=np.uint8),
        "classification_probability": np.asarray(probabilities, dtype=np.float32),
        "positive_patch_id": np.asarray(["b"]),
        "segmentation_thresholds": np.asarray([0.4, 0.5], dtype=np.float32),
        "segmentation_dice_grid": np.asarray([[dice, dice + 0.1]], dtype=np.float32),
    }


class ComposeCandidateRefinerValidationTests(unittest.TestCase):
    def test_composition_uses_confidence_and_contour_from_declared_sources(self):
        classification = archive(probabilities=(0.1, 0.9), dice=0.2)
        contour = archive(probabilities=(0.3, 0.7), dice=0.6)
        result = compose(classification, contour)
        np.testing.assert_array_equal(
            result["classification_probability"],
            classification["classification_probability"],
        )
        np.testing.assert_array_equal(
            result["segmentation_dice_grid"], contour["segmentation_dice_grid"]
        )

    def test_identity_or_label_mismatch_fails_closed(self):
        classification = archive()
        contour = archive()
        contour["patch_id"] = np.asarray(["b", "a"])
        with self.assertRaisesRegex(ValueError, "patch_id"):
            compose(classification, contour)
        contour = archive()
        contour["label"] = np.asarray([1, 1], dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "label"):
            compose(classification, contour)

    def test_incomplete_contour_grid_fails_closed(self):
        classification = archive()
        contour = archive()
        contour["segmentation_dice_grid"] = np.zeros((2, 2), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "Dice-grid"):
            compose(classification, contour)


if __name__ == "__main__":
    unittest.main()
