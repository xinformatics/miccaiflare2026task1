import unittest

import numpy as np

from generate_lung_blob_proposals import blob_candidates, paint_sphere


class LungBlobProposalTests(unittest.TestCase):
    def test_bright_spherical_nodule_is_ranked_near_center(self):
        shape = (48, 48, 48)
        ct = np.full(shape, -800.0, dtype=np.float32)
        center = np.asarray([24.0, 24.0, 24.0])
        grid = np.indices(shape, dtype=np.float32)
        distance = np.sqrt(sum((grid[axis] - center[axis]) ** 2 for axis in range(3)))
        ct[distance < 12] = 50.0  # body shell enclosing synthetic lung air
        ct[distance < 9] = -800.0
        nodule = np.sqrt(sum((grid[axis] - center[axis]) ** 2 for axis in range(3))) < 2.5
        ct[nodule] = 100.0
        candidates = blob_candidates(
            ct, np.ones(3), 1.0, (2.0, 3.0), 0.01, 4.0, 20
        )
        self.assertTrue(candidates)
        nearest = min(np.linalg.norm(coordinate - center) for _, coordinate in candidates)
        self.assertLessEqual(nearest, 3.0)

    def test_painted_sphere_is_binary_and_local(self):
        mask = np.zeros((20, 20, 20), dtype=bool)
        paint_sphere(mask, np.asarray([10.0, 10.0, 10.0]), np.ones(3), 3.0)
        self.assertTrue(mask[10, 10, 10])
        self.assertFalse(mask[0, 0, 0])
        self.assertGreater(np.count_nonzero(mask), 1)


if __name__ == "__main__":
    unittest.main()
