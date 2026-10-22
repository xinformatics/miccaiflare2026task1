import unittest

import numpy as np

from generate_lung_grid_proposals import lung_grid_centers


class LungGridProposalTests(unittest.TestCase):
    def test_grid_centers_cover_occupied_synthetic_lung_cells(self):
        shape = (64, 64, 64)
        ct = np.full(shape, -800.0, dtype=np.float32)
        grid = np.indices(shape, dtype=np.float32)
        center = np.asarray([32.0, 32.0, 32.0])
        distance = np.sqrt(sum((grid[axis] - center[axis]) ** 2 for axis in range(3)))
        ct[distance < 24] = 50.0
        ct[distance < 18] = -800.0
        centers = lung_grid_centers(ct, np.ones(3), 1.0, 16.0)
        self.assertGreater(len(centers), 1)
        self.assertTrue(all(center.shape == (3,) for center in centers))


if __name__ == "__main__":
    unittest.main()
