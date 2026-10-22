import sys
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
from unittest import mock


PANCANCER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PANCANCER))

from predict_stunet_component_crops import (  # noqa: E402
    component_crop_boxes,
    cropped_affine,
    decompress_nifti_gzip,
    save_crop,
)


class ComponentCropInferenceTest(unittest.TestCase):
    def test_component_boxes_expand_clip_and_merge(self):
        anchor = np.zeros((20, 30, 40), dtype=bool)
        anchor[0:2, 4:6, 7:9] = True
        anchor[5:7, 4:6, 7:9] = True
        boxes = component_crop_boxes(anchor, (2.0, 2.0, 2.0), 4.0)
        self.assertEqual(boxes, [(slice(0, 9), slice(2, 8), slice(5, 11))])

    def test_disjoint_components_remain_separate(self):
        anchor = np.zeros((30, 30, 30), dtype=bool)
        anchor[2:4, 2:4, 2:4] = True
        anchor[20:22, 20:22, 20:22] = True
        boxes = component_crop_boxes(anchor, (1.0, 1.0, 1.0), 2.0)
        self.assertEqual(len(boxes), 2)

    def test_crop_affine_preserves_world_coordinates(self):
        affine = np.array([[2, 0, 0, 10], [0, 3, 0, 20], [0, 0, 4, 30], [0, 0, 0, 1]], dtype=float)
        box = (slice(3, 8), slice(4, 9), slice(5, 10))
        observed = cropped_affine(affine, box) @ np.array([0, 0, 0, 1])
        expected = affine @ np.array([3, 4, 5, 1])
        np.testing.assert_allclose(observed, expected)

    def test_saved_crop_geometry_and_values(self):
        data = np.arange(10 * 11 * 12, dtype=np.int16).reshape(10, 11, 12)
        affine = np.diag([2.0, 3.0, 4.0, 1.0])
        reference = nib.Nifti1Image(data, affine)
        box = (slice(2, 7), slice(3, 8), slice(4, 9))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "crop.nii.gz"
            save_crop(path, data[box], reference, box)
            crop = nib.load(str(path))
            np.testing.assert_array_equal(np.asanyarray(crop.dataobj), data[box])
            np.testing.assert_allclose(crop.affine, cropped_affine(affine, box))

    def test_source_has_no_hidden_label_or_upload_interface(self):
        source = (PANCANCER / "predict_stunet_component_crops.py").read_text(encoding="utf-8")
        self.assertNotIn('add_argument("--labels"', source)
        for forbidden in ("Validation-Hidden-Labels", "codabench.org", "requests.post", "curl ", "selenium"):
            self.assertNotIn(forbidden, source)

    def test_parallel_decompress_contract_is_argument_vector_not_shell(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.nii.gz"
            destination = root / "destination.nii"
            source.write_bytes(b"compressed")
            with mock.patch("predict_stunet_component_crops.shutil.which", return_value="/usr/bin/pigz"), mock.patch(
                "predict_stunet_component_crops.subprocess.run"
            ) as run:
                decompress_nifti_gzip(source, destination, 6)
            command = run.call_args.args[0]
            self.assertEqual(command, ["/usr/bin/pigz", "-p", "6", "-dc", str(source)])
            self.assertTrue(run.call_args.kwargs["check"])
            self.assertNotIn("shell", run.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
