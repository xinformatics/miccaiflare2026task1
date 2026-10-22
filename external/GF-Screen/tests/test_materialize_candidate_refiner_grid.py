import argparse
import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from apply_candidate_refiner import local_target_grid, materialize
from materialize_candidate_refiner_grid import materialize_grid


class GridMaterializerParityTests(unittest.TestCase):
    def test_per_configuration_segmentation_thresholds_share_one_case_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            predictions = work / "predictions"
            patches = work / "patches"
            proposals = root / "proposals"
            base = root / "base"
            for directory in (predictions, patches, proposals, base):
                directory.mkdir(parents=True, exist_ok=True)
            shape = (16, 16, 16)
            affine = np.eye(4)
            image_path = root / "image.nii.gz"
            nib.save(nib.Nifti1Image(np.zeros(shape, np.float32), affine), image_path)
            seed = np.zeros(shape, np.uint8)
            seed[6:10, 6:10, 6:10] = 1
            nib.save(nib.Nifti1Image(seed, affine), proposals / "case.nii.gz")
            nib.save(nib.Nifti1Image(np.zeros(shape, np.uint8), affine), base / "case.nii.gz")
            probability = np.zeros((1,) + shape, np.float16)
            probability[0, 5:11, 5:11, 5:11] = 0.8
            np.savez_compressed(
                predictions / "patch.npz",
                classification_probability=np.asarray(0.8, np.float32),
                segmentation_probability=probability,
            )
            np.savez_compressed(patches / "patch.npz", patch_affine=affine)
            manifest = {
                "component_policy": "union",
                "tile_stride_mm": 72.0,
                "source_names": ["seed"],
                "case_records": [{
                    "case": "case", "image": str(image_path),
                    "source_proposals": [str(proposals / "case.nii.gz")],
                }],
                "records": [{
                    "case": "case", "component_id": 1, "source_name": "seed",
                    "patch_id": "patch", "tile_index": 0,
                    "patch": str(patches / "patch.npz"),
                    "seed_mm3": float(seed.sum()),
                }],
            }
            (work / "manifest.json").write_text(json.dumps(manifest))
            configs = [
                {
                    "tag": "s050", "classification_threshold": 0.5,
                    "segmentation_threshold": 0.5, "mode": "refined",
                },
                {
                    "tag": "s090", "classification_threshold": 0.5,
                    "segmentation_threshold": 0.9, "mode": "refined",
                },
            ]
            config_path = root / "configs.json"
            config_path.write_text(json.dumps({"configurations": configs}))
            grid_root = root / "grid"
            materialize_grid(
                work, base, grid_root, config_path, 0, 0.7, True,
                0, 0, 0, 1e12,
            )
            low = np.asanyarray(
                nib.load(grid_root / "s050" / "fold0" / "case.nii.gz").dataobj
            ).astype(bool)
            high = np.asanyarray(
                nib.load(grid_root / "s090" / "fold0" / "case.nii.gz").dataobj
            ).astype(bool)
            self.assertGreater(int(low.sum()), 0)
            self.assertEqual(int(high.sum()), 0)
            for tag, expected in (("s050", 0.5), ("s090", 0.9)):
                marker = json.loads(
                    (grid_root / tag / "fold0" / "CANDIDATE_REFINER_COMPLETE.json").read_text()
                )
                self.assertEqual(marker["segmentation_threshold"], expected)
                self.assertEqual(marker["fallback_segmentation_threshold"], 0.7)
                reference = root / "reference" / tag
                materialize(argparse.Namespace(
                    work=work, base=base, output=reference,
                    source_classification_threshold=[],
                    source_segmentation_threshold=[], source_mode=[],
                    classification_threshold=0.5,
                    segmentation_threshold=expected, mode="refined",
                    empty_base=True, min_seed_coverage=0,
                    min_refined_overlap=0, min_volume_ratio=0,
                    max_volume_ratio=1e12, min_base_dice=0.25,
                    min_base_seed_coverage=0.5, case_limit=0,
                ))
                expected_mask = np.asanyarray(
                    nib.load(reference / "case.nii.gz").dataobj
                ).astype(bool)
                observed_mask = np.asanyarray(
                    nib.load(grid_root / tag / "fold0" / "case.nii.gz").dataobj
                ).astype(bool)
                np.testing.assert_array_equal(observed_mask, expected_mask)

    def test_refined_or_seed_falls_back_only_when_geometry_is_unstable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            predictions = work / "predictions"
            patches = work / "patches"
            proposals = root / "proposals"
            base = root / "base"
            for directory in (predictions, patches, proposals, base):
                directory.mkdir(parents=True, exist_ok=True)
            shape = (16, 16, 16)
            affine = np.eye(4)
            image_path = root / "image.nii.gz"
            nib.save(nib.Nifti1Image(np.zeros(shape, np.float32), affine), image_path)
            seed = np.zeros(shape, np.uint8)
            seed[2:5, 2:5, 2:5] = 1
            refined = np.zeros((1,) + shape, np.float16)
            refined[0, 11:14, 11:14, 11:14] = 0.9
            nib.save(nib.Nifti1Image(seed, affine), proposals / "case.nii.gz")
            nib.save(
                nib.Nifti1Image(np.zeros(shape, np.uint8), affine),
                base / "case.nii.gz",
            )
            np.savez_compressed(
                predictions / "patch.npz",
                classification_probability=np.asarray(0.8, np.float32),
                segmentation_probability=refined,
            )
            np.savez_compressed(patches / "patch.npz", patch_affine=affine)
            manifest = {
                "component_policy": "union",
                "tile_stride_mm": 72.0,
                "source_names": ["seed"],
                "case_records": [{
                    "case": "case", "image": str(image_path),
                    "source_proposals": [str(proposals / "case.nii.gz")],
                }],
                "records": [{
                    "case": "case", "component_id": 1, "source_name": "seed",
                    "patch_id": "patch", "tile_index": 0,
                    "patch": str(patches / "patch.npz"),
                    "seed_mm3": float(seed.sum()),
                }],
            }
            (work / "manifest.json").write_text(json.dumps(manifest))
            configs = [{
                "tag": "fallback", "classification_threshold": 0.5,
                "mode": "refined_or_seed", "source_classification_thresholds": {},
            }]
            config_path = root / "configs.json"
            config_path.write_text(json.dumps({"configurations": configs}))
            grid_root = root / "grid"
            materialize_grid(
                work, base, grid_root, config_path, 0, 0.5, True,
                0.1, 0.1, 0.25, 4.0,
            )
            reference = root / "reference"
            materialize(argparse.Namespace(
                work=work, base=base, output=reference,
                source_classification_threshold=[],
                source_segmentation_threshold=[], source_mode=[],
                classification_threshold=0.5, segmentation_threshold=0.5,
                mode="refined_or_seed", empty_base=True,
                min_seed_coverage=0.1, min_refined_overlap=0.1,
                min_volume_ratio=0.25, max_volume_ratio=4.0,
                min_base_dice=0.25, min_base_seed_coverage=0.5, case_limit=0,
            ))
            observed = np.asanyarray(
                nib.load(grid_root / "fallback" / "fold0" / "case.nii.gz").dataobj
            ).astype(bool)
            reference_mask = np.asanyarray(
                nib.load(reference / "case.nii.gz").dataobj
            ).astype(bool)
            np.testing.assert_array_equal(observed, seed.astype(bool))
            np.testing.assert_array_equal(reference_mask, observed)
            audit = (grid_root / "fallback" / "fold0.components.csv").read_text()
            self.assertIn("refinement_accepted", audit)
            self.assertIn(",0,", audit)

    def test_prune_base_removes_only_fully_rejected_source_component(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            predictions = work / "predictions"
            patches = work / "patches"
            proposals = root / "proposals"
            base = root / "base"
            for directory in (predictions, patches, proposals, base):
                directory.mkdir(parents=True, exist_ok=True)
            shape = (16, 16, 16)
            affine = np.eye(4)
            image_path = root / "image.nii.gz"
            nib.save(nib.Nifti1Image(np.zeros(shape, np.float32), affine), image_path)
            components = []
            union = np.zeros(shape, np.uint8)
            for bounds in ((2, 5), (11, 14)):
                component = np.zeros(shape, np.uint8)
                component[bounds[0] : bounds[1], 6:9, 6:9] = 1
                components.append(component)
                union |= component
            nib.save(nib.Nifti1Image(union, affine), proposals / "case.nii.gz")
            nib.save(nib.Nifti1Image(union, affine), base / "case.nii.gz")
            records = []
            for component_id, probability in ((1, 0.8), (2, 0.2)):
                patch_id = f"patch{component_id}"
                np.savez_compressed(
                    predictions / f"{patch_id}.npz",
                    classification_probability=np.asarray(probability, np.float32),
                    segmentation_probability=np.zeros((1,) + shape, np.float16),
                )
                np.savez_compressed(patches / f"{patch_id}.npz", patch_affine=affine)
                records.append({
                    "case": "case", "component_id": component_id,
                    "source_name": "a", "patch_id": patch_id, "tile_index": 0,
                    "patch": str(patches / f"{patch_id}.npz"),
                    "seed_mm3": float(components[component_id - 1].sum()),
                })
            manifest = {
                "component_policy": "source_separate", "tile_stride_mm": 72.0,
                "source_names": ["a"],
                "case_records": [{
                    "case": "case", "image": str(image_path),
                    "source_proposals": [str(proposals / "case.nii.gz")],
                }],
                "records": records,
            }
            (work / "manifest.json").write_text(json.dumps(manifest))
            config = {
                "tag": "prune", "classification_threshold": 0.5,
                "mode": "refined", "source_classification_thresholds": {},
                "source_segmentation_thresholds": {},
                "source_deployment_modes": {"a": "prune_base"},
            }
            config_path = root / "configs.json"
            config_path.write_text(json.dumps({"configurations": [config]}))
            grid_root = root / "grid"
            materialize_grid(
                work, base, grid_root, config_path, 0, 0.5, False, 0, 0, 0, 1e12
            )
            reference = root / "reference"
            materialize(argparse.Namespace(
                work=work, base=base, output=reference,
                source_classification_threshold=[], source_segmentation_threshold=[],
                source_mode=["a=prune_base"], classification_threshold=0.5,
                segmentation_threshold=0.5, mode="refined", empty_base=False,
                min_seed_coverage=0, min_refined_overlap=0, min_volume_ratio=0,
                max_volume_ratio=1e12, min_base_dice=0.25,
                min_base_seed_coverage=0.5, case_limit=0,
            ))
            observed = np.asanyarray(
                nib.load(grid_root / "prune" / "fold0" / "case.nii.gz").dataobj
            ).astype(bool)
            reference_mask = np.asanyarray(
                nib.load(reference / "case.nii.gz").dataobj
            ).astype(bool)
            np.testing.assert_array_equal(observed, components[0].astype(bool))
            np.testing.assert_array_equal(reference_mask, observed)
            with self.assertRaisesRegex(ValueError, "nonempty base"):
                materialize_grid(
                    work, base, root / "invalid", config_path, 0, 0.5,
                    True, 0, 0, 0, 1e12,
                )

    def test_source_seed_crop_keeps_only_approved_tile_footprint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            predictions = work / "predictions"
            patches = work / "patches"
            proposals = root / "proposals"
            base = root / "base"
            for directory in (predictions, patches, proposals, base):
                directory.mkdir(parents=True, exist_ok=True)
            shape = (16, 16, 16)
            affine = np.eye(4)
            image_path = root / "image.nii.gz"
            nib.save(nib.Nifti1Image(np.zeros(shape, np.float32), affine), image_path)
            seed = np.zeros(shape, np.uint8)
            seed[2:12, 6:8, 6:8] = 1
            nib.save(nib.Nifti1Image(seed, affine), proposals / "case.nii.gz")
            nib.save(nib.Nifti1Image(np.zeros(shape, np.uint8), affine), base / "case.nii.gz")
            probability = np.zeros((1, 8, 8, 8), np.float16)
            np.savez_compressed(
                predictions / "patch.npz",
                classification_probability=np.asarray(0.8, np.float32),
                segmentation_probability=probability,
            )
            np.savez_compressed(patches / "patch.npz", patch_affine=affine)
            manifest = {
                "component_policy": "source_separate",
                "tile_stride_mm": 72.0,
                "source_names": ["a"],
                "case_records": [{
                    "case": "case", "image": str(image_path),
                    "source_proposals": [str(proposals / "case.nii.gz")],
                }],
                "records": [{
                    "case": "case", "component_id": 1, "source_name": "a",
                    "patch_id": "patch", "tile_index": 0,
                    "patch": str(patches / "patch.npz"), "seed_mm3": float(seed.sum()),
                }],
            }
            (work / "manifest.json").write_text(json.dumps(manifest))
            config = {
                "tag": "seed_crop", "classification_threshold": 0.5,
                "mode": "refined", "source_classification_thresholds": {},
                "source_segmentation_thresholds": {},
                "source_deployment_modes": {"a": "seed_crop"},
            }
            config_path = root / "configs.json"
            config_path.write_text(json.dumps({"configurations": [config]}))
            grid_root = root / "grid"
            materialize_grid(
                work, base, grid_root, config_path, 0, 0.5, True, 0, 0, 0, 1e12
            )
            reference = root / "reference"
            materialize(argparse.Namespace(
                work=work, base=base, output=reference,
                source_classification_threshold=[], source_segmentation_threshold=[],
                source_mode=["a=seed_crop"], classification_threshold=0.5,
                segmentation_threshold=0.5, mode="refined", empty_base=True,
                min_seed_coverage=0, min_refined_overlap=0, min_volume_ratio=0,
                max_volume_ratio=1e12, min_base_dice=0.25,
                min_base_seed_coverage=0.5, case_limit=0,
            ))
            observed = np.asanyarray(
                nib.load(grid_root / "seed_crop" / "fold0" / "case.nii.gz").dataobj
            ).astype(bool)
            reference_mask = np.asanyarray(
                nib.load(reference / "case.nii.gz").dataobj
            ).astype(bool)
            slices, _, _ = local_target_grid((8, 8, 8), affine, shape, affine)
            expected = np.zeros(shape, dtype=bool)
            expected[slices] = seed[slices].astype(bool)
            np.testing.assert_array_equal(observed, expected)
            np.testing.assert_array_equal(reference_mask, expected)
            self.assertLess(int(observed.sum()), int(seed.sum()))

    def test_matches_reference_materializer_for_three_modes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            predictions = work / "predictions"
            patches = work / "patches"
            proposals = root / "proposals"
            base = root / "base"
            for directory in (predictions, patches, proposals, base):
                directory.mkdir(parents=True, exist_ok=True)
            shape = (16, 16, 16)
            affine = np.eye(4)
            image_path = root / "image.nii.gz"
            nib.save(nib.Nifti1Image(np.zeros(shape, np.float32), affine), image_path)
            seed = np.zeros(shape, np.uint8)
            seed[6:10, 6:10, 6:10] = 1
            nib.save(nib.Nifti1Image(seed, affine), proposals / "case.nii.gz")
            nib.save(nib.Nifti1Image(np.zeros(shape, np.uint8), affine), base / "case.nii.gz")
            probability = np.zeros((1,) + shape, np.float16)
            probability[0, 5:11, 5:11, 5:11] = 0.9
            np.savez_compressed(
                predictions / "patch.npz",
                classification_probability=np.asarray(0.8, np.float32),
                segmentation_probability=probability,
            )
            np.savez_compressed(patches / "patch.npz", patch_affine=affine)
            manifest = {
                "component_policy": "union",
                "tile_stride_mm": 72.0,
                "source_names": ["seed"],
                "case_records": [
                    {
                        "case": "case",
                        "image": str(image_path),
                        "source_proposals": [str(proposals / "case.nii.gz")],
                    }
                ],
                "records": [
                    {
                        "case": "case",
                        "component_id": 1,
                        "source_name": "seed",
                        "patch_id": "patch",
                        "tile_index": 0,
                        "patch": str(patches / "patch.npz"),
                        "seed_mm3": float(seed.sum()),
                    }
                ],
            }
            (work / "manifest.json").write_text(json.dumps(manifest))
            configs = []
            for mode in ("refined", "seed", "seed_union"):
                configs.append(
                    {
                        "tag": f"c050_{mode}",
                        "classification_threshold": 0.5,
                        "mode": mode,
                        "source_classification_thresholds": {},
                    }
                )
            config_path = root / "configs.json"
            config_path.write_text(json.dumps({"configurations": configs}))
            grid_root = root / "grid"
            materialize_grid(
                work, base, grid_root, config_path, 0, 0.5, True, 0, 0, 0, 1e12
            )
            for config in configs:
                reference = root / "reference" / config["tag"]
                args = argparse.Namespace(
                    work=work,
                    base=base,
                    output=reference,
                    source_classification_threshold=None,
                    classification_threshold=0.5,
                    segmentation_threshold=0.5,
                    mode=config["mode"],
                    empty_base=True,
                    min_seed_coverage=0,
                    min_refined_overlap=0,
                    min_volume_ratio=0,
                    max_volume_ratio=1e12,
                    min_base_dice=0.25,
                    min_base_seed_coverage=0.5,
                    case_limit=0,
                )
                materialize(args)
                expected = np.asanyarray(nib.load(reference / "case.nii.gz").dataobj)
                observed = np.asanyarray(
                    nib.load(grid_root / config["tag"] / "fold0" / "case.nii.gz").dataobj
                )
                np.testing.assert_array_equal(observed, expected)

    def test_matches_source_separated_thresholds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / "work"
            predictions = work / "predictions"
            patches = work / "patches"
            base = root / "base"
            source_a, source_b = root / "source_a", root / "source_b"
            for directory in (predictions, patches, base, source_a, source_b):
                directory.mkdir(parents=True, exist_ok=True)
            shape = (16, 16, 16)
            affine = np.eye(4)
            image_path = root / "image.nii.gz"
            nib.save(nib.Nifti1Image(np.zeros(shape, np.float32), affine), image_path)
            seeds = []
            for directory, bounds in ((source_a, (2, 5)), (source_b, (11, 14))):
                seed = np.zeros(shape, np.uint8)
                seed[bounds[0] : bounds[1], 6:9, 6:9] = 1
                seeds.append(seed)
                nib.save(nib.Nifti1Image(seed, affine), directory / "case.nii.gz")
            nib.save(nib.Nifti1Image(np.zeros(shape, np.uint8), affine), base / "case.nii.gz")
            records = []
            for index, source_name in enumerate(("a", "b"), 1):
                patch_id = f"patch{index}"
                probability = np.zeros((1,) + shape, np.float16)
                probability[0, seeds[index - 1].astype(bool)] = 0.9
                if source_name == "a":
                    probability[0, 1, 6, 6] = 0.9
                np.savez_compressed(
                    predictions / f"{patch_id}.npz",
                    classification_probability=np.asarray(0.6, np.float32),
                    segmentation_probability=probability,
                )
                np.savez_compressed(patches / f"{patch_id}.npz", patch_affine=affine)
                records.append(
                    {
                        "case": "case",
                        "component_id": index,
                        "source_name": source_name,
                        "patch_id": patch_id,
                        "tile_index": 0,
                        "patch": str(patches / f"{patch_id}.npz"),
                        "seed_mm3": float(seeds[index - 1].sum()),
                    }
                )
            manifest = {
                "component_policy": "source_separate",
                "tile_stride_mm": 72.0,
                "source_names": ["a", "b"],
                "case_records": [
                    {
                        "case": "case",
                        "image": str(image_path),
                        "source_proposals": [
                            str(source_a / "case.nii.gz"),
                            str(source_b / "case.nii.gz"),
                        ],
                    }
                ],
                "records": records,
            }
            (work / "manifest.json").write_text(json.dumps(manifest))
            config = {
                "tag": "source_calibrated",
                "classification_threshold": 0.5,
                "mode": "refined",
                "source_classification_thresholds": {"a": 0.5, "b": 0.7},
                "source_segmentation_thresholds": {"a": 0.45, "b": 0.75},
                "source_deployment_modes": {"a": "seed"},
            }
            config_path = root / "configs.json"
            config_path.write_text(json.dumps({"configurations": [config]}))
            grid_root = root / "grid"
            materialize_grid(
                work, base, grid_root, config_path, 0, 0.5, True, 0, 0, 0, 1e12
            )
            reference = root / "reference"
            args = argparse.Namespace(
                work=work,
                base=base,
                output=reference,
                source_classification_threshold=["a=0.5", "b=0.7"],
                source_segmentation_threshold=["a=0.45", "b=0.75"],
                source_mode=["a=seed"],
                classification_threshold=0.5,
                segmentation_threshold=0.5,
                mode="refined",
                empty_base=True,
                min_seed_coverage=0,
                min_refined_overlap=0,
                min_volume_ratio=0,
                max_volume_ratio=1e12,
                min_base_dice=0.25,
                min_base_seed_coverage=0.5,
                case_limit=0,
            )
            materialize(args)
            expected = np.asanyarray(nib.load(reference / "case.nii.gz").dataobj)
            observed = np.asanyarray(
                nib.load(grid_root / config["tag"] / "fold0" / "case.nii.gz").dataobj
            )
            np.testing.assert_array_equal(observed, expected)
            np.testing.assert_array_equal(observed, seeds[0])


if __name__ == "__main__":
    unittest.main()
