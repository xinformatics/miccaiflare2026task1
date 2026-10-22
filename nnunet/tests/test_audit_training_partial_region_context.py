import tempfile
import unittest
from pathlib import Path

import numpy as np

from nnunet.audit_training_partial_region_context import (
    PATCH_SHAPE,
    crop_centered,
    inventory_sha256,
    patch_policy_metrics,
    select_source_cases,
    trusted_mask,
)


class TrainingPartialRegionContextAuditTest(unittest.TestCase):
    def test_source_selection_is_sorted_evenly_spaced_and_stable(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for source, count in (("A", 7), ("B", 3)):
                for index in range(count):
                    (folder / f"{source}_{index:03d}_seg.b2nd").touch()
            selected, counts = select_source_cases(folder, per_source=4)
        self.assertEqual(counts, {"A": 7, "B": 3})
        self.assertEqual(selected, ["A_000", "A_002", "A_004", "A_006", "B_000", "B_001", "B_002"])
        self.assertEqual(inventory_sha256(selected), inventory_sha256(selected))

    def test_source_selection_excludes_known_empty_cases_before_spacing(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for index in range(6):
                (folder / f"A_{index:03d}_seg.b2nd").touch()
            selected, counts = select_source_cases(folder, per_source=3, excluded={"A_002"})
        self.assertEqual(counts, {"A": 5})
        self.assertEqual(selected, ["A_000", "A_003", "A_005"])

    def test_center_crop_pads_with_unknown_and_keeps_center(self):
        segmentation = np.zeros((20, 30, 40), dtype=np.int16)
        segmentation[0, 0, 0] = 1
        patch = crop_centered(segmentation, np.asarray([0, 0, 0]))
        self.assertEqual(patch.shape, PATCH_SHAPE)
        self.assertEqual(int((patch == 1).sum()), 1)
        self.assertGreater(int((patch == -1).sum()), 0)

    def test_shape_band_is_subset_of_matching_box_and_retains_foreground(self):
        foreground = np.zeros((18, 22, 26), dtype=bool)
        foreground[4:14, 8:10, 9:11] = True
        foreground[12:14, 8:18, 9:11] = True
        margins = (2, 3, 4)
        band = trusted_mask(foreground, "band", margins)
        box = trusted_mask(foreground, "box", margins)
        self.assertTrue(np.all(band <= box))
        self.assertTrue(np.all(band[foreground]))
        self.assertLess(int(band.sum()), int(box.sum()))

    def test_unknown_voxels_are_not_counted_as_background(self):
        segmentation = np.full((12, 14, 16), -1, dtype=np.int16)
        segmentation[3:9, 3:11, 3:13] = 0
        segmentation[5:7, 6:8, 7:9] = 1
        metrics = patch_policy_metrics(segmentation, "band", (3, 3, 3))
        self.assertEqual(metrics["foreground_voxels"], 8)
        self.assertEqual(metrics["retained_foreground_voxels"], 8)
        self.assertGreater(metrics["trusted_unknown_voxels"], 0)
        self.assertLess(metrics["trusted_background_voxels"], int((segmentation == 0).sum()))


if __name__ == "__main__":
    unittest.main()
