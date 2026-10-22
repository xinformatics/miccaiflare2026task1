import unittest

import numpy as np

from analyze_uls_feature_classifier_cv import Candidate, CaseStats
from analyze_uls_ranked_selector_cv import macro_dice_selected, ranked_selection


def candidate(case: str, component_id: int) -> Candidate:
    return Candidate(
        row={
            "case": case,
            "component_id": component_id,
            "crop_id": f"{case}__c{component_id:04d}",
        },
        addition_indices=np.asarray([component_id], dtype=np.int64),
    )


class RankedSelectorTests(unittest.TestCase):
    def test_cap_is_applied_independently_per_case(self):
        candidates = [
            candidate("a", 1),
            candidate("a", 2),
            candidate("a", 3),
            candidate("b", 1),
            candidate("b", 2),
        ]
        probabilities = np.asarray([0.7, 0.9, 0.8, 0.6, 0.95])
        selected = ranked_selection(candidates, probabilities, 0.5, 2, {"a", "b"})
        self.assertEqual(selected.tolist(), [False, True, True, True, True])

    def test_zero_cap_means_unlimited_and_threshold_is_inclusive(self):
        candidates = [candidate("a", 1), candidate("a", 2), candidate("a", 3)]
        probabilities = np.asarray([0.49, 0.50, 0.90])
        selected = ranked_selection(candidates, probabilities, 0.5, 0, {"a"})
        self.assertEqual(selected.tolist(), [False, True, True])

    def test_invalid_lengths_and_negative_cap_fail(self):
        candidates = [candidate("a", 1)]
        with self.assertRaises(ValueError):
            ranked_selection(candidates, np.asarray([]), 0.5, 1, {"a"})
        with self.assertRaises(ValueError):
            ranked_selection(candidates, np.asarray([0.9]), 0.5, -1, {"a"})

    def test_macro_dice_unions_overlapping_additions_exactly(self):
        candidates = [candidate("a", 1), candidate("a", 2)]
        candidates[0].addition_indices = np.asarray([100, 102], dtype=np.int64)
        candidates[1].addition_indices = np.asarray([100, 103], dtype=np.int64)
        stats = {
            "a": CaseStats(
                shape=(1, 1, 200),
                base_voxels=10,
                target_voxels=10,
                base_overlap=5,
                target_indices=np.asarray([100, 101], dtype=np.int64),
                reference=None,
            )
        }
        score = macro_dice_selected(
            candidates, np.asarray([True, True]), {"a"}, stats
        )
        self.assertAlmostEqual(score, 12.0 / 23.0)


if __name__ == "__main__":
    unittest.main()
