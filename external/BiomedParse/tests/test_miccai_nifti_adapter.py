import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from miccai_nifti_adapter import case_name, convert_case, prediction_name, prepare_case, window_to_uint8


class AdapterTests(unittest.TestCase):
    def test_window_to_uint8_clips_and_rounds(self) -> None:
        image = np.asarray([-200.0, -160.0, 40.0, 240.0, 300.0])
        self.assertEqual(window_to_uint8(image, -160.0, 240.0).tolist(), [0, 0, 128, 255, 255])

    def test_invalid_window_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            window_to_uint8(np.zeros(1), 1.0, 1.0)

    def test_case_name_handles_double_extension(self) -> None:
        self.assertEqual(case_name(Path("FLARE23Ts_0001.nii.gz")), "FLARE23Ts_0001")
        self.assertEqual(
            prediction_name(Path("FLARE23Ts_0001_0000.nii.gz")),
            "FLARE23Ts_0001.nii.gz",
        )

    def test_round_trip_preserves_geometry_and_binarizes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            array = np.arange(3 * 4 * 5, dtype=np.int16).reshape(3, 4, 5) - 20
            image = sitk.GetImageFromArray(array)
            image.SetSpacing((0.8, 1.1, 2.7))
            image.SetOrigin((3.0, -2.0, 8.0))
            image.SetDirection((0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0))
            source = root / "case.nii.gz"
            sitk.WriteImage(image, str(source))

            prepared = root / "case.npz"
            prepare_case(source, prepared, ["Tumor", "Malignant lesion"], -160, 240)
            with np.load(prepared, allow_pickle=True) as archive:
                self.assertEqual(archive["imgs"].dtype, np.uint8)
                self.assertEqual(archive["imgs"].shape, array.shape)
                self.assertEqual(
                    archive["text_prompts"].item(),
                    {"1": "Tumor", "2": "Malignant lesion", "instance_label": 0},
                )

            raw = np.zeros_like(array, dtype=np.uint8)
            raw[1, 2, 3] = 2
            prediction = root / "prediction.npz"
            np.savez_compressed(prediction, segs=raw)
            destination = root / "prediction.nii.gz"
            convert_case(source, prediction, destination)
            result = sitk.ReadImage(str(destination))
            np.testing.assert_allclose(result.GetSpacing(), image.GetSpacing(), rtol=0, atol=1e-6)
            np.testing.assert_allclose(result.GetOrigin(), image.GetOrigin(), rtol=0, atol=1e-6)
            np.testing.assert_allclose(result.GetDirection(), image.GetDirection(), rtol=0, atol=1e-6)
            self.assertEqual(int(sitk.GetArrayFromImage(result).sum()), 1)

    def test_convert_rejects_shape_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = sitk.GetImageFromArray(np.zeros((3, 4, 5), dtype=np.int16))
            source = root / "case.nii.gz"
            sitk.WriteImage(image, str(source))
            prediction = root / "prediction.npz"
            np.savez_compressed(prediction, segs=np.zeros((3, 4, 4), dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, "shape"):
                convert_case(source, prediction, root / "out.nii.gz")

    def test_convert_selects_requested_segmentation_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = sitk.GetImageFromArray(np.zeros((2, 3, 4), dtype=np.int16))
            source = root / "case.nii.gz"
            sitk.WriteImage(image, str(source))
            prediction = root / "prediction.npz"
            low = np.zeros((2, 3, 4), dtype=np.uint8)
            high = np.zeros_like(low)
            low[0, 0, 0] = 1
            high[1, 2, 3] = 1
            np.savez_compressed(prediction, segs_e030=low, segs_e070=high)
            destination = root / "prediction.nii.gz"
            convert_case(
                source, prediction, destination, segmentation_key="segs_e070"
            )
            result = sitk.GetArrayFromImage(sitk.ReadImage(str(destination)))
            self.assertEqual(tuple(np.argwhere(result)[0]), (1, 2, 3))


if __name__ == "__main__":
    unittest.main()
