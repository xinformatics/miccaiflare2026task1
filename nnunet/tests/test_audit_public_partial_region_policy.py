import unittest

import numpy as np

from nnunet.audit_public_partial_box_collision import scenario_metrics_from_slice
from nnunet.audit_public_partial_region_policy import (
    expanded_slice,
    shape_band_metrics_from_slice,
)


class PartialRegionPolicyTest(unittest.TestCase):
    def test_expanded_slice_uses_physical_axis_margins(self):
        region, margins = expanded_slice(
            (slice(1, 3), slice(5, 7), slice(8, 10)),
            (12, 15, 20),
            (4.0, 2.0, 1.0),
            4.0,
        )
        self.assertEqual(tuple(margins), (1, 2, 4))
        self.assertEqual(region, (slice(0, 4), slice(3, 9), slice(4, 14)))

    def test_shape_band_excludes_distant_bbox_corner(self):
        labels = np.zeros((7, 10, 10), dtype=np.uint8)
        labels[3, 2, 2:7] = 1
        labels[3, 2:7, 6] = 1
        labels[3, 6, 2] = 2
        component_slice = (slice(3, 4), slice(2, 7), slice(2, 7))
        foreground = int((labels > 0).sum())
        box = scenario_metrics_from_slice(
            labels, 1, component_slice, 9, foreground, (1.0, 1.0, 1.0), 1.0
        )
        band = shape_band_metrics_from_slice(
            labels, 1, component_slice, 9, foreground, (1.0, 1.0, 1.0), 1.0
        )
        self.assertTrue(box["any_collision"])
        self.assertFalse(band["any_collision"])
        self.assertLess(
            band["trusted_background_to_primary_ratio"],
            box["trusted_background_to_primary_ratio"],
        )

    def test_shape_band_detects_neighbor_within_margin(self):
        labels = np.zeros((7, 7, 7), dtype=np.uint8)
        labels[3, 3, 3] = 1
        labels[3, 3, 5] = 2
        row = shape_band_metrics_from_slice(
            labels,
            1,
            (slice(3, 4), slice(3, 4), slice(3, 4)),
            1,
            2,
            (1.0, 1.0, 1.0),
            2.0,
        )
        self.assertTrue(row["any_collision"])
        self.assertEqual(row["omitted_component_ids_inside_box"], [2])


if __name__ == "__main__":
    unittest.main()
