import tempfile
import unittest
import sys
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np

PANCANCER = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PANCANCER))

from predict_stunet_safe import (
    compact_probability_exports,
    expected_output_name,
    reusable_compact_outputs,
    resolve_trainer,
    safe_initialize,
    sequential_image_lists,
    validate_tile_step_size,
    validate_shard_outputs,
)
from STUNetTrainerPanCancer import (
    STUNetTrainerPanCancer,
    STUNetTrainerPanCancerComponentBox,
    STUNetTrainerPanCancerComponentBoxE160Balanced,
)
from STUNetTrainerPanCancerCompleteHybrid import (
    STUNetTrainerPanCancerComponentBoxE160CompleteHybrid,
)
from STUNetTrainerPanCancerSourceBalanced import (
    STUNetTrainerPanCancerComponentBoxE160SourceStratified,
)


class PredictSTUNetSafeTest(unittest.TestCase):
    def test_safe_initialize_falls_back_for_torch_without_weights_only(self):
        calls = []

        def legacy_load(path, map_location=None):
            calls.append((path, map_location))
            return {"network_weights": {}}

        class FakePredictor:
            def initialize_from_trained_model_folder(
                self, model_dir, use_folds, checkpoint_name
            ):
                import torch

                torch.load("bundled.pth", map_location="cpu")
                calls.append((model_dir, use_folds, checkpoint_name))

        with mock.patch("predict_stunet_safe.torch.load", legacy_load):
            safe_initialize(FakePredictor(), Path("model"), "checkpoint.pth")
        self.assertEqual(calls[0], ("bundled.pth", "cpu"))
        self.assertEqual(calls[1], ("model", ("all",), "checkpoint.pth"))

    def test_custom_trainer_resolver_is_exact(self):
        self.assertIs(resolve_trainer(None, "STUNetTrainerPanCancer", None), STUNetTrainerPanCancer)

    def test_output_name_contract(self):
        self.assertEqual(expected_output_name(Path("FLARE-Val_0001_0000.nii.gz")), "FLARE-Val_0001.nii.gz")
        with self.assertRaisesRegex(ValueError, "invalid challenge"):
            expected_output_name(Path("FLARE-Val_0001.nii.gz"))

    def test_sequential_shard_selection_matches_strided_contract(self):
        inventory = [Path(f"case_{index}_0000.nii.gz") for index in range(7)]
        self.assertEqual(
            sequential_image_lists(inventory, 1, 3),
            [["case_1_0000.nii.gz"], ["case_4_0000.nii.gz"]],
        )

    def test_tile_step_validation_preserves_baseline_and_bounds_experiments(self):
        self.assertEqual(validate_tile_step_size(0.5), 0.5)
        self.assertEqual(validate_tile_step_size(1), 1.0)
        for invalid in (0, -0.1, 1.01):
            with self.assertRaisesRegex(ValueError, "tile step size"):
                validate_tile_step_size(invalid)

    def test_shard_validation_checks_geometry_binary_and_disjoint_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary);images = root / "images";output = root / "output"
            images.mkdir();output.mkdir();affine=np.diag([1.2,1.3,2.4,1.0])
            inventory=[]
            for index in range(4):
                source=images/f"FLARE-Val_{index:04d}_0000.nii.gz"
                nib.save(nib.Nifti1Image(np.zeros((3,4,5),dtype=np.int16),affine),source)
                inventory.append(source)
            for source in inventory[1::2]:
                destination=output/expected_output_name(source)
                nib.save(nib.Nifti1Image(np.zeros((3,4,5),dtype=np.uint8),affine),destination)
            cases=validate_shard_outputs(inventory,output,part_id=1,num_parts=2)
            self.assertEqual(cases,["FLARE-Val_0001.nii.gz","FLARE-Val_0003.nii.gz"])

    def test_source_exposes_no_label_argument_or_upload(self):
        source = Path(__file__).resolve().parents[1] / "predict_stunet_safe.py"
        text = source.read_text(encoding="utf-8")
        self.assertNotIn('add_argument("--labels"', text)

    def test_component_box_trainer_is_an_explicit_safe_inference_class(self):
        source = PANCANCER / "predict_stunet_safe.py"
        text = source.read_text(encoding="utf-8")
        self.assertIs(
            resolve_trainer(None, "STUNetTrainerPanCancerComponentBox", None),
            STUNetTrainerPanCancerComponentBox,
        )
        self.assertIs(
            resolve_trainer(
                None, "STUNetTrainerPanCancerComponentBoxE160Balanced", None
            ),
            STUNetTrainerPanCancerComponentBoxE160Balanced,
        )
        self.assertIs(
            resolve_trainer(
                None,
                "STUNetTrainerPanCancerComponentBoxE160CompleteHybrid",
                None,
            ),
            STUNetTrainerPanCancerComponentBoxE160CompleteHybrid,
        )
        self.assertIs(
            resolve_trainer(
                None,
                "STUNetTrainerPanCancerComponentBoxE160SourceStratified",
                None,
            ),
            STUNetTrainerPanCancerComponentBoxE160SourceStratified,
        )
        self.assertIn("STUNetTrainerPanCancerComponentBox", text)
        self.assertIn("TRAINER_CLASSES", text)
        self.assertIn("trainer_name not in TRAINER_CLASSES", text)
        self.assertIn("reusable_compact_outputs", text)
        self.assertIn("predict_from_files_sequential", text)
        self.assertIn('default=0.5', text)
        self.assertIn('"tile_step_size": args.tile_step_size', text)
        self.assertIn('export_kwargs["num_threads_torch"] = args.workers', text)
        self.assertIn("torch.set_num_interop_threads(1)", text)
        self.assertIn('kwargs["weights_only"] = True', text)
        self.assertIn('"uploaded": False', text)
        for forbidden in ("codabench.org", "requests.post", "curl ", "selenium"):
            self.assertNotIn(forbidden, text)

    def test_probability_compaction_uses_nifti_order_and_removes_raw_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            name = "FLARE-Val_0001.nii.gz"
            shape = (3, 4, 5)
            lesion_zyx = np.arange(np.prod(shape), dtype=np.float32).reshape(shape[::-1])
            lesion_zyx /= lesion_zyx.max()
            expected_binary = (lesion_zyx.transpose(2, 1, 0) > 0.5).astype(np.uint8)
            nib.save(
                nib.Nifti1Image(expected_binary, np.eye(4)),
                output / name,
            )
            probabilities = np.stack((1.0 - lesion_zyx, lesion_zyx))
            np.savez_compressed(output / "FLARE-Val_0001.npz", probabilities=probabilities)
            (output / "FLARE-Val_0001.pkl").write_bytes(b"temporary")
            files = compact_probability_exports([name], output)
            with np.load(output / files[0]) as archive:
                compact = archive["lesion_probability_uint16"]
            expected = np.rint(lesion_zyx.transpose(2, 1, 0) * 65535).astype(np.uint16)
            self.assertTrue(np.array_equal(compact, expected))
            self.assertFalse((output / "FLARE-Val_0001.npz").exists())
            self.assertFalse((output / "FLARE-Val_0001.pkl").exists())

    def test_compact_mask_pair_is_reusable_without_raw_probability(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            name = "FLARE-Val_0001.nii.gz"
            nib.save(
                nib.Nifti1Image(np.zeros((3, 4, 5), dtype=np.uint8), np.eye(4)),
                output / name,
            )
            np.savez_compressed(
                output / "FLARE-Val_0001.prob16.npz",
                lesion_probability_uint16=np.zeros((3, 4, 5), dtype=np.uint16),
            )
            self.assertTrue(reusable_compact_outputs([name], output))
            (output / "FLARE-Val_0001.npz").write_bytes(b"partial raw")
            (output / "FLARE-Val_0001.pkl").write_bytes(b"partial properties")
            self.assertTrue(reusable_compact_outputs([name], output))
            self.assertEqual(compact_probability_exports([name], output), ["FLARE-Val_0001.prob16.npz"])
            self.assertFalse((output / "FLARE-Val_0001.npz").exists())
            self.assertFalse((output / "FLARE-Val_0001.pkl").exists())


if __name__ == "__main__":
    unittest.main()
