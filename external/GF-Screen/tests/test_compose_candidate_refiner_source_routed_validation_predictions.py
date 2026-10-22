import unittest

import numpy as np

from compose_candidate_refiner_source_routed_validation_predictions import (
    compose_source_routed,
)


def archive(probabilities=(0.2, 0.8, 0.7), rows=((0.1, 0.2), (0.3, 0.4))):
    return {
        "patch_id": np.asarray(["a", "b", "c"]),
        "label": np.asarray([0, 1, 1], dtype=np.uint8),
        "classification_probability": np.asarray(probabilities, dtype=np.float32),
        "positive_patch_id": np.asarray(["b", "c"]),
        "segmentation_thresholds": np.asarray([0.4, 0.5], dtype=np.float32),
        "segmentation_dice_grid": np.asarray(rows, dtype=np.float32),
    }


class SourceRoutedCompositionTests(unittest.TestCase):
    def test_routes_positive_contours_and_preserves_classifier(self):
        classification = archive(probabilities=(0.1, 0.9, 0.8))
        mature = archive(rows=((0.7, 0.8), (0.1, 0.2)))
        discovery = archive(rows=((0.3, 0.4), (0.8, 0.9)))
        result, counts = compose_source_routed(
            classification,
            {"mature": mature, "discovery": discovery},
            {"a": "epoch105", "b": "currentgf", "c": "lung_cell0"},
            {
                "epoch105": "mature",
                "currentgf": "mature",
                "lung_cell0": "discovery",
            },
        )
        np.testing.assert_array_equal(
            result["classification_probability"],
            classification["classification_probability"],
        )
        np.testing.assert_array_equal(
            result["segmentation_dice_grid"],
            np.asarray(((0.7, 0.8), (0.8, 0.9)), dtype=np.float32),
        )
        self.assertEqual(counts, {"mature": 1, "discovery": 1})

    def test_held_out_archive_may_be_subset_of_full_index(self):
        result, _ = compose_source_routed(
            archive(),
            {"mature": archive(), "discovery": archive()},
            {
                "a": "epoch105",
                "b": "currentgf",
                "c": "lung_cell0",
                "held_in_other_fold": "autopet_weak",
            },
            {
                "epoch105": "mature",
                "currentgf": "mature",
                "lung_cell0": "discovery",
                "autopet_weak": "discovery",
            },
        )
        self.assertEqual(len(result["patch_id"]), 3)

    def test_missing_source_route_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "source-route coverage"):
            compose_source_routed(
                archive(),
                {"mature": archive(), "discovery": archive()},
                {"a": "epoch105", "b": "currentgf", "c": "lung_cell0"},
                {"epoch105": "mature", "currentgf": "mature"},
            )

    def test_identity_mismatch_fails_closed(self):
        bad = archive()
        bad["positive_patch_id"] = np.asarray(["c", "b"])
        with self.assertRaisesRegex(ValueError, "positive_patch_id"):
            compose_source_routed(
                archive(),
                {"mature": archive(), "discovery": bad},
                {"a": "epoch105", "b": "currentgf", "c": "lung_cell0"},
                {
                    "epoch105": "mature",
                    "currentgf": "mature",
                    "lung_cell0": "discovery",
                },
            )


if __name__ == "__main__":
    unittest.main()
