import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from render_nested_teacher_montage import (
    _proxy_slice,
    _stream_mask_indices,
    render,
    selected_slices,
)


class NoFullReadProxy:
    """Array proxy that fails if production code requests the full volume."""

    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        if item is Ellipsis or item == ():
            raise AssertionError("full-volume proxy read")
        return self.data[item]

    def __array__(self, *args, **kwargs):
        raise AssertionError("full-volume proxy conversion")


class NestedTeacherMontageTests(unittest.TestCase):
    def test_selected_slices_stay_inside_foreground_extent(self):
        mask = np.zeros((12, 14, 16), dtype=bool)
        mask[2:10, 3:12, 4:14] = True
        for axis, bounds in enumerate(((2, 9), (3, 11), (4, 13))):
            values = selected_slices(mask, axis)
            self.assertTrue(values)
            self.assertTrue(all(bounds[0] <= value <= bounds[1] for value in values))

    def test_render_writes_png_and_rejects_non_nested_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            affine = np.diag([1.0, 1.0, 2.0, 1.0])
            image = np.zeros((12, 14, 16), dtype=np.float32)
            support = np.zeros_like(image, dtype=np.uint8)
            support[2:10, 3:12, 4:14] = 1
            seed = np.zeros_like(support)
            seed[4:8, 5:10, 6:12] = 1
            paths = [root / name for name in ("image.nii.gz", "support.nii.gz", "seed.nii.gz")]
            for path, data in zip(paths, (image, support, seed)):
                nib.save(nib.Nifti1Image(data, affine), path)
            output = root / "montage.png"
            render(*paths, output)
            self.assertGreater(output.stat().st_size, 1000)

            seed[0, 0, 0] = 1
            nib.save(nib.Nifti1Image(seed, affine), paths[2])
            with self.assertRaisesRegex(ValueError, "not nested"):
                render(*paths, output)

    def test_streaming_helpers_never_materialize_full_volume(self):
        support = np.zeros((20, 22, 35), dtype=np.uint8)
        support[2:18, 4:20, 3:34] = 1
        seed = np.zeros_like(support)
        seed[7:13, 9:15, 12:24] = 1
        support_proxy = NoFullReadProxy(support)
        seed_proxy = NoFullReadProxy(seed)
        indices = _stream_mask_indices(
            support_proxy, seed_proxy, support.shape, slab_depth=7
        )
        self.assertEqual(set(indices), {0, 1, 2})
        self.assertTrue(all(2 <= value <= 17 for value in indices[0]))
        self.assertTrue(all(4 <= value <= 19 for value in indices[1]))
        self.assertTrue(all(3 <= value <= 33 for value in indices[2]))
        plane = _proxy_slice(support_proxy, 2, indices[2][0])
        self.assertEqual(plane.ndim, 2)


if __name__ == "__main__":
    unittest.main()
