import unittest
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analyze_component_classifier_cv import (
    ANATOMY_FEATURES,
    FEATURES,
    CaseStats,
    extract_anatomy_features,
    macro_dice,
    nested_group_predictions,
)


class ComponentClassifierCVTests(unittest.TestCase):
    def test_anatomy_features_capture_family_overlap_and_physical_distance(self):
        component_map = np.zeros((5, 5, 5), dtype=np.int32)
        component_map[1, 1, 1] = 1
        component_map[4, 4, 4] = 2
        counts = np.bincount(component_map.ravel(), minlength=3)
        anatomy = np.zeros_like(component_map)
        anatomy[1, 1, 1] = 5  # liver / solid-organ family
        values = extract_anatomy_features(
            component_map, counts, anatomy, np.asarray([1.0, 1.0, 2.0])
        )
        self.assertEqual(set(values), set(ANATOMY_FEATURES))
        self.assertEqual(values["anatomy_solid_organ_fraction"][1], 1.0)
        self.assertEqual(values["anatomy_nonzero_fraction"][2], 0.0)
        self.assertGreater(values["anatomy_distance_min_log1p_mm"][2], 0.0)

    def test_macro_dice_uses_only_selected_disconnected_components(self):
        rows = [
            {"case": "a", "component_id": 1},
            {"case": "a", "component_id": 2},
        ]
        stats = {
            "a": CaseStats(
                base_voxels=2,
                target_voxels=4,
                base_overlap=2,
                component_voxels={1: 2, 2: 3},
                component_overlap={1: 2, 2: 0},
            )
        }
        probabilities = np.asarray([0.9, 0.8])
        self.assertEqual(macro_dice(rows, probabilities, 0.85, {"a"}, stats), 1.0)
        self.assertAlmostEqual(
            macro_dice(rows, probabilities, 0.5, {"a"}, stats), 8.0 / 11.0
        )

    def test_nested_group_predictions_cover_every_component(self):
        rows = []
        stats = {}
        for case_index in range(10):
            case = f"case_{case_index:02d}"
            stats[case] = CaseStats(
                base_voxels=1,
                target_voxels=2,
                base_overlap=1,
                component_voxels={1: 1, 2: 1},
                component_overlap={1: 1, 2: 0},
            )
            for component_id, label in ((1, 1), (2, 0)):
                row = {
                    "case": case,
                    "component_id": component_id,
                    "is_true_component": label,
                }
                row.update({name: 0.0 for name in FEATURES})
                row["log_volume_mm3"] = float(label)
                rows.append(row)

        tiny_model = make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced")
        )
        with patch(
            "analyze_component_classifier_cv.model_grid",
            return_value=[("tiny_logistic", tiny_model)],
        ):
            probabilities, folds = nested_group_predictions(rows, stats, seed=7)
        self.assertEqual(len(folds), 5)
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(all("outer_fold" in row for row in rows))
        self.assertTrue(all("oof_threshold" in row for row in rows))


if __name__ == "__main__":
    unittest.main()
