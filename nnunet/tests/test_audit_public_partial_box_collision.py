import unittest

import numpy as np

from nnunet.audit_public_partial_box_collision import (
    expanded_component_box,
    scenario_metrics,
    scenario_metrics_from_slice,
    summarize,
)


class PartialBoxCollisionTest(unittest.TestCase):
    def test_physical_margin_is_anisotropic_and_clipped(self):
        component = np.zeros((8, 12, 12), dtype=bool)
        component[1, 5, 5] = True
        box = expanded_component_box(component, (4.0, 2.0, 1.0), 4.0)
        self.assertEqual(box, (slice(0, 3), slice(3, 8), slice(1, 10)))

    def test_neighbor_collision_is_detected(self):
        labels = np.zeros((9, 9, 9), dtype=np.uint8)
        labels[4, 4, 4] = 1
        labels[4, 4, 6] = 2
        near = scenario_metrics(labels, 1, (1.0, 1.0, 1.0), 2.0)
        far = scenario_metrics(labels, 1, (1.0, 1.0, 1.0), 1.0)
        self.assertTrue(near["any_collision"])
        self.assertEqual(near["omitted_voxels_inside_box"], 1)
        self.assertFalse(far["any_collision"])

    def test_slice_implementation_matches_reference(self):
        labels = np.zeros((9, 9, 9), dtype=np.uint8)
        labels[3:5, 3:5, 3:5] = 1
        labels[4:6, 4:6, 7:9] = 2
        reference = scenario_metrics(labels, 1, (2.0, 1.0, 1.0), 3.0)
        efficient = scenario_metrics_from_slice(
            labels,
            1,
            (slice(3, 5), slice(3, 5), slice(3, 5)),
            8,
            16,
            (2.0, 1.0, 1.0),
            3.0,
        )
        self.assertEqual(reference, efficient)

    def test_summary_conditions_on_scenarios_with_omissions(self):
        rows = [
            {
                "any_collision": False,
                "omitted_voxels": 0,
                "omitted_voxels_inside_box": 0,
                "trusted_background_to_primary_ratio": 4.0,
            },
            {
                "any_collision": True,
                "omitted_voxels": 10,
                "omitted_voxels_inside_box": 2,
                "trusted_background_to_primary_ratio": 8.0,
            },
        ]
        result = summarize(rows)
        self.assertEqual(result["scenarios_with_omitted_lesions"], 1)
        self.assertEqual(result["collision_rate_when_omitted_lesions_exist"], 1.0)
        self.assertAlmostEqual(result["pooled_omitted_voxel_collision_fraction"], 0.2)
        self.assertEqual(result["trusted_background_to_primary_ratio"]["median"], 6.0)


if __name__ == "__main__":
    unittest.main()
