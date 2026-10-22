import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np

from prepare_healthy_hard_negatives import (
    build_crops,
    centered_crop,
    component_candidates,
    load_healthy_json,
    stage_images,
)


class HealthyHardNegativeTests(unittest.TestCase):
    def test_evaluation_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = root / "validation"
            evaluation.mkdir()
            item = {
                "image": str(evaluation / "case_0000.nii.gz"),
                "label": str(evaluation / "case.nii.gz"),
                "name": "case",
            }
            source_json = root / "healthy.json"
            source_json.write_text(
                json.dumps({"training": [item], "validation": []}), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "evaluation data is forbidden"):
                load_healthy_json(source_json)

    def test_stage_directory_contains_only_nifti_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source_0000.nii.gz"
            source.write_bytes(b"synthetic-nifti-placeholder")
            label = root / "source.nii.gz"
            label.write_bytes(b"unused")
            source_json = root / "healthy.json"
            item = {"image": str(source), "label": str(label), "name": "source"}
            source_json.write_text(
                json.dumps({"training": [item], "validation": [item]}), encoding="utf-8"
            )
            staging = root / "staged_images"

            stage_images(source_json, staging)

            self.assertEqual([path.name for path in staging.iterdir()], [source.name])
            self.assertTrue((staging / source.name).is_symlink())
            manifest_path = root / "staged_images_manifest.json"
            self.assertTrue(manifest_path.is_file())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["items"][0]["staged"], str((staging / source.name).absolute()))

    def test_centered_crop_preserves_center_and_zero_pads_edges(self):
        volume = np.arange(5 * 6 * 7, dtype=np.int32).reshape(5, 6, 7)
        crop, start = centered_crop(volume, center=(0, 1, 2), roi=(4, 4, 4))

        self.assertEqual(start, (-2, -1, 0))
        self.assertEqual(crop.shape, (4, 4, 4))
        np.testing.assert_array_equal(crop[2:, 1:, :], volume[:2, :3, :4])
        self.assertFalse(bool(crop[:2].any()))
        self.assertFalse(bool(crop[:, 0].any()))

    def test_component_candidates_use_physical_volume_and_largest_first(self):
        mask = np.zeros((12, 12, 12), dtype=np.uint8)
        mask[1:3, 1:3, 1:3] = 1  # 8 voxels * 3 mm3 = 24 mm3
        mask[5:8, 5:8, 5:8] = 1  # 27 voxels * 3 mm3 = 81 mm3
        mask[9:11, 9:11, 9:11] = 1  # another 24 mm3 component

        candidates = component_candidates(mask, voxel_mm3=3.0, min_mm3=20.0, max_components=2)

        self.assertEqual([item["voxels"] for item in candidates], [27, 8])
        self.assertEqual([item["volume_mm3"] for item in candidates], [81.0, 24.0])
        self.assertEqual(candidates[0]["center"], (6, 6, 6))

    def test_component_candidates_zero_max_keeps_all_and_threshold_is_inclusive(self):
        mask = np.zeros((8, 8, 8), dtype=np.uint8)
        mask[1:3, 1:3, 1:3] = 1
        mask[5:7, 5:7, 5:7] = 1

        candidates = component_candidates(mask, voxel_mm3=2.0, min_mm3=16.0, max_components=0)

        self.assertEqual(len(candidates), 2)
        self.assertTrue(all(item["volume_mm3"] == 16.0 for item in candidates))

    def test_build_emits_zero_label_crop_and_balanced_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "Healthy_0000.nii.gz"
            label_path = root / "Healthy.nii.gz"
            predictions = root / "predictions"
            predictions.mkdir()
            affine = np.diag([1.0, 1.0, 3.0, 1.0])
            image = np.full((16, 16, 16), 50.0, dtype=np.float32)
            label = np.zeros_like(image, dtype=np.uint8)
            prediction = np.zeros_like(image, dtype=np.uint8)
            prediction[5:8, 5:8, 5:8] = 1
            nib.save(nib.Nifti1Image(image, affine), image_path)
            nib.save(nib.Nifti1Image(label, affine), label_path)
            nib.save(nib.Nifti1Image(prediction, affine), predictions / "Healthy.nii.gz")
            source_json = root / "healthy.json"
            item = {"image": str(image_path), "label": str(label_path), "name": "Healthy"}
            source_json.write_text(
                json.dumps({"training": [item], "validation": [item]}), encoding="utf-8"
            )
            output = root / "output"
            output_json = root / "derived.json"
            build_crops(
                SimpleNamespace(
                    healthy_json=source_json,
                    predictions=predictions,
                    output=output,
                    output_json=output_json,
                    spacing=(1.0, 1.0, 3.0),
                    roi=(8, 8, 8),
                    min_mm3=16.0,
                    max_components_per_case=4,
                    reference_healthy_repeat=100,
                )
            )

            derived = json.loads(output_json.read_text(encoding="utf-8"))
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(derived["training"]), 2)
            self.assertEqual(manifest["hard_negative_crops"], 1)
            self.assertEqual(manifest["recommended_healthy_repeat"], 50)
            crop_item = derived["training"][1]
            crop_image = np.asanyarray(nib.load(crop_item["image"]).dataobj)
            crop_label = np.asanyarray(nib.load(crop_item["label"]).dataobj)
            self.assertEqual(crop_image.shape, (8, 8, 8))
            self.assertFalse(bool(crop_label.any()))

    def test_build_rejects_nonempty_source_label(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "NotHealthy_0000.nii.gz"
            label_path = root / "NotHealthy.nii.gz"
            predictions = root / "predictions"
            predictions.mkdir()
            affine = np.diag([1.0, 1.0, 3.0, 1.0])
            image = np.full((12, 12, 12), 50.0, dtype=np.float32)
            label = np.zeros_like(image, dtype=np.uint8)
            label[4:6, 4:6, 4:6] = 1
            prediction = label.copy()
            nib.save(nib.Nifti1Image(image, affine), image_path)
            nib.save(nib.Nifti1Image(label, affine), label_path)
            nib.save(nib.Nifti1Image(prediction, affine), predictions / "NotHealthy.nii.gz")
            source_json = root / "healthy.json"
            item = {"image": str(image_path), "label": str(label_path), "name": "NotHealthy"}
            source_json.write_text(
                json.dumps({"training": [item], "validation": [item]}), encoding="utf-8"
            )

            with self.assertRaisesRegex(ValueError, "not label-confirmed healthy"):
                build_crops(
                    SimpleNamespace(
                        healthy_json=source_json,
                        predictions=predictions,
                        output=root / "output",
                        output_json=root / "derived.json",
                        spacing=(1.0, 1.0, 3.0),
                        roi=(8, 8, 8),
                        min_mm3=16.0,
                        max_components_per_case=4,
                        reference_healthy_repeat=100,
                    )
                )


if __name__ == "__main__":
    unittest.main()
