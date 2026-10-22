import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

PANCANCER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PANCANCER))
from compact_probability_export import (
    compact_lesion_probability,
    export_prediction_from_logits_compact,
)


class _Writer:
    def write_seg(self, segmentation, destination, properties):
        nib.save(nib.Nifti1Image(segmentation, np.eye(4)), destination)


class _Plans:
    image_reader_writer_class = _Writer

    def get_label_manager(self, dataset_json):
        return object()


class CompactProbabilityExportTest(unittest.TestCase):
    def test_transposes_probability_to_nifti_order_and_writes_exact_uint16(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shape = (3, 4, 5)
            lesion_zyx = np.linspace(0, 1, np.prod(shape), dtype=np.float32).reshape(
                shape[::-1]
            )
            segmentation = (lesion_zyx.transpose(2, 1, 0) > 0.5).astype(np.uint8)
            nifti = root / "case.nii.gz"
            nib.save(nib.Nifti1Image(segmentation, np.eye(4)), nifti)
            probabilities = np.stack((1.0 - lesion_zyx, lesion_zyx))
            output = compact_lesion_probability(probabilities, nifti, root / "case")
            with np.load(output) as archive:
                observed = archive["lesion_probability_uint16"]
            expected = np.rint(lesion_zyx.transpose(2, 1, 0) * 65535).astype(np.uint16)
            self.assertTrue(np.array_equal(observed, expected))
            self.assertFalse((root / "case.npz").exists())

    def test_threshold_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nifti = root / "case.nii.gz"
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.uint8), np.eye(4)), nifti)
            probabilities = np.zeros((2, 2, 2, 2), dtype=np.float32)
            probabilities[0] = 1.0
            with self.assertRaisesRegex(ValueError, "threshold/orientation"):
                compact_lesion_probability(probabilities, nifti, root / "case")

    def test_drop_in_callback_accepts_nnunet_string_output_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shape = (3, 4, 5)
            lesion = np.linspace(0, 1, np.prod(shape), dtype=np.float32).reshape(
                shape
            )
            segmentation = (lesion > 0.5).astype(np.uint8)
            probabilities = np.stack((1.0 - lesion, lesion))
            with patch(
                "compact_probability_export.convert_predicted_logits_to_segmentation_with_correct_shape",
                return_value=(segmentation, probabilities),
            ):
                export_prediction_from_logits_compact(
                    np.zeros((2, *shape), dtype=np.float32),
                    {},
                    object(),
                    _Plans(),
                    {"file_ending": ".nii.gz"},
                    str(root / "case"),
                    save_probabilities=True,
                )
            self.assertTrue((root / "case.nii.gz").is_file())
            with np.load(root / "case.prob16.npz") as archive:
                restored = archive["lesion_probability_uint16"]
            self.assertEqual(restored.dtype, np.uint16)
            self.assertEqual(restored.shape, shape)
            self.assertTrue(np.array_equal(restored > 32768, segmentation.astype(bool)))


if __name__ == "__main__":
    unittest.main()
