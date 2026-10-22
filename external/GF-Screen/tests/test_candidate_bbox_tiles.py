import unittest

import numpy as np

from candidate_bbox_tiles import (
    axis_centers,
    component_world_bounds,
    in_crop_target,
    tile_centers_world,
)


class CandidateBoundingBoxTileTests(unittest.TestCase):
    def test_axis_centers_cover_long_extent_with_bounded_stride(self):
        centers = axis_centers(0.0, 200.0, 96.0, 72.0)
        np.testing.assert_allclose(centers, [48.0, 100.0, 152.0])
        self.assertLessEqual(float(np.diff(centers).max()), 72.0)
        self.assertLessEqual(centers[0] - 48.0, 0.0)
        self.assertGreaterEqual(centers[-1] + 48.0, 200.0)

    def test_world_bounds_include_voxel_cells_and_margin(self):
        low, high = component_world_bounds(
            np.asarray([[2, 3, 4], [3, 5, 4]]), np.eye(4), 2.0
        )
        np.testing.assert_allclose(low, [-0.5, 0.5, 1.5])
        np.testing.assert_allclose(high, [5.5, 7.5, 6.5])

    def test_tile_grid_cartesian_product(self):
        centers = tile_centers_world(
            np.zeros(3), np.asarray([200.0, 20.0, 20.0]), (64, 64, 64),
            (1.5, 1.5, 1.5), 72.0
        )
        self.assertEqual(len(centers), 3)
        np.testing.assert_allclose([value[0] for value in centers], [48, 100, 152])

    def test_subvoxel_fallback_only_for_target_inside_tile(self):
        target = np.zeros((8, 8, 8), dtype=bool)
        target[2, 2, 2] = True
        empty = np.zeros((4, 4, 4), dtype=bool)
        inside, used = in_crop_target(target, np.eye(4), np.eye(4), empty.shape, empty)
        self.assertTrue(used)
        self.assertTrue(inside[2, 2, 2])

        shifted = np.eye(4)
        shifted[:3, 3] = 20.0
        outside, used = in_crop_target(target, np.eye(4), shifted, empty.shape, empty)
        self.assertFalse(used)
        self.assertFalse(outside.any())


if __name__ == "__main__":
    unittest.main()
