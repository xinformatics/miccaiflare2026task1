import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference_naming import incomplete_inputs, prediction_name


class InferenceNamingTests(unittest.TestCase):
    def test_nnunet_channel_suffix_is_removed(self):
        self.assertEqual(
            prediction_name("FLARE-Val_0001_0000.nii.gz"),
            "FLARE-Val_0001.nii.gz",
        )

    def test_auxiliary_name_is_not_truncated(self):
        self.assertEqual(
            prediction_name("train_2670_a_1.nii.gz"),
            "train_2670_a_1.nii.gz",
        )

    def test_healthy_names_remain_unique(self):
        names = [
            "train_3422_a_1.nii.gz",
            "train_3427_a_1.nii.gz",
            "CHAOS_01_0000.nii.gz",
        ]
        outputs = [prediction_name(name) for name in names]
        self.assertEqual(len(outputs), len(set(outputs)))

    def test_rejects_non_nifti(self):
        with self.assertRaises(ValueError):
            prediction_name("scan.npz")

    def test_incomplete_inputs_filters_only_outputs_complete_in_every_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = [root / "a_0000.nii.gz", root / "b_0000.nii.gz"]
            outputs = [root / "threshold_a", root / "threshold_b"]
            for directory in outputs:
                directory.mkdir()
                (directory / "a.nii.gz").touch()
            (outputs[0] / "b.nii.gz").touch()
            self.assertEqual(incomplete_inputs(inputs, outputs), [inputs[1]])

    def test_incomplete_inputs_requires_an_output_root(self):
        with self.assertRaisesRegex(ValueError, "prediction directory"):
            incomplete_inputs([Path("case_0000.nii.gz")], [])


if __name__ == "__main__":
    unittest.main()
