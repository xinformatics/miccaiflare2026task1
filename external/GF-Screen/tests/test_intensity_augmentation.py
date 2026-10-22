import unittest
from types import SimpleNamespace

import torch

from utils.data_utils import (
    RemapClippedWindowd,
    appearance_augmentation_transforms,
)


class IntensityAugmentationTests(unittest.TestCase):
    def test_cached_wide_window_maps_to_soft_window(self):
        transform = RemapClippedWindowd(keys="image")
        hu = torch.tensor([-900.0, -175.0, 37.5, 250.0, 650.0])
        wide_scaled = (hu + 900.0) / 1550.0
        actual = transform({"image": wide_scaled})["image"]
        expected = torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0])
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_appearance_augmentations_are_opt_in(self):
        disabled = SimpleNamespace(
            enable_scale_intensity_aug=False,
            enable_contrast_aug=False,
        )
        self.assertEqual(appearance_augmentation_transforms(disabled), [])

        enabled = SimpleNamespace(
            enable_scale_intensity_aug=True,
            RandScaleIntensityd_prob=0.1,
            enable_contrast_aug=True,
            RandAdjustContrastd_prob=0.2,
            contrast_gamma_min=0.7,
            contrast_gamma_max=1.5,
        )
        transforms = appearance_augmentation_transforms(enabled)
        self.assertEqual(len(transforms), 2)
        self.assertEqual(transforms[0].__class__.__name__, "RandScaleIntensityd")
        self.assertEqual(transforms[1].__class__.__name__, "RandAdjustContrastd")


if __name__ == "__main__":
    unittest.main()
