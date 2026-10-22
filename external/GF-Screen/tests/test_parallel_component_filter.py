import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from fuse_and_filter_masks import filter_components, process_predictions


class ParallelComponentFilterTests(unittest.TestCase):
    def test_26_connectivity_can_rescue_diagonal_fragments(self):
        mask = np.zeros((8, 8, 8), dtype=bool)
        mask[1:3, 1:3, 1:3] = True
        mask[3:5, 3:5, 3:5] = True
        six = filter_components(mask, voxel_volume=1.0, min_mm3=12.0, connectivity=6)
        twenty_six = filter_components(
            mask, voxel_volume=1.0, min_mm3=12.0, connectivity=26
        )
        self.assertFalse(six.any())
        np.testing.assert_array_equal(twenty_six, mask)

    def test_default_connectivity_preserves_historical_six(self):
        mask = np.zeros((8, 8, 8), dtype=bool)
        mask[1:3, 1:3, 1:3] = True
        mask[3:5, 3:5, 3:5] = True
        implicit = filter_components(mask, voxel_volume=1.0, min_mm3=12.0)
        explicit = filter_components(
            mask, voxel_volume=1.0, min_mm3=12.0, connectivity=6
        )
        np.testing.assert_array_equal(implicit, explicit)

    def test_parallel_matches_sequential_exactly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            sequential = root / "sequential"
            parallel = root / "parallel"
            source.mkdir()
            sequential.mkdir()
            parallel.mkdir()
            names = []
            affine = np.diag([2.0, 2.0, 2.0, 1.0])
            for index in range(6):
                mask = np.zeros((16, 15, 14), dtype=np.uint8)
                mask[1:3, 1:3, 1:3] = 1  # 64 mm3: removed
                mask[5:10, 5:10, 5:10] = 1  # 1,000 mm3: retained
                if index % 2:
                    mask[11:15, 9:14, 8:13] = 1
                name = f"case_{index:02d}.nii.gz"
                names.append(name)
                nib.save(nib.Nifti1Image(mask, affine), source / name)

            process_predictions(
                names, source, None, "a", 1000, sequential, workers=1
            )
            process_predictions(
                names, source, None, "a", 1000, parallel, workers=4
            )
            for name in names:
                expected = np.asanyarray(nib.load(sequential / name).dataobj)
                actual = np.asanyarray(nib.load(parallel / name).dataobj)
                np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
