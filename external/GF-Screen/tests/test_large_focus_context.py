import unittest

import torch

from models.glance import recenter_crops_3d_tensor


class LargeFocusContextTests(unittest.TestCase):
    def test_recenters_and_clamps_at_volume_boundaries(self):
        image = torch.arange(1 * 200 * 180 * 96).reshape(1, 200, 180, 96)
        crops, coords = recenter_crops_3d_tensor(
            image,
            [(0, 0, 0), (104, 84, 32)],
            crop_size=(128, 128, 64),
        )
        self.assertEqual(crops.shape, (2, 1, 128, 128, 64))
        self.assertEqual(coords, [(0, 0, 0), (72, 52, 32)])
        self.assertTrue(torch.equal(crops[0], image[:, :128, :128, :64]))
        self.assertTrue(torch.equal(crops[1], image[:, 72:200, 52:180, 32:96]))

    def test_deduplicates_expanded_starts_stably(self):
        image = torch.zeros(1, 160, 160, 64)
        crops, coords = recenter_crops_3d_tensor(
            image,
            [(0, 0, 0), (24, 24, 0), (64, 64, 0)],
            crop_size=(128, 128, 64),
        )
        self.assertEqual(coords, [(0, 0, 0), (8, 8, 0), (32, 32, 0)])
        self.assertEqual(crops.shape[0], 3)

    def test_rejects_context_larger_than_volume(self):
        with self.assertRaisesRegex(ValueError, "larger than input"):
            recenter_crops_3d_tensor(
                torch.zeros(1, 96, 96, 64),
                [(0, 0, 0)],
                crop_size=(128, 128, 64),
            )


if __name__ == "__main__":
    unittest.main()
