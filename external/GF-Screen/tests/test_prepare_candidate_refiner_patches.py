import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from audit_candidate_refiner_patches import audit_row
from prepare_candidate_refiner_patches import (
    component_records,
    crop_affine,
    deterministic_negative_subset,
    pad_slices,
    resample_array,
    resample_binary_target,
    resample_seed_prompt,
    slice_affine,
    process_case,
    load_resumable_case,
    source_separated_component_records,
    source_separated_component_records_compact,
    unpack_local_component,
)


class CandidateRefinerPatchTests(unittest.TestCase):
    def test_positive_only_contract_never_materializes_unmatched_proposals(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proposals = root / "proposals"
            output = root / "output"
            proposals.mkdir()
            affine = np.eye(4)
            ct = np.full((24, 24, 24), 30.0, dtype=np.float32)
            target = np.zeros_like(ct, dtype=np.uint8)
            target[5:9, 5:9, 5:9] = 1
            proposal = np.zeros_like(target)
            proposal[6:8, 6:8, 6:8] = 1
            proposal[16:19, 16:19, 16:19] = 1
            image_path = root / "case_0000.nii.gz"
            label_path = root / "case.nii.gz"
            nib.save(nib.Nifti1Image(ct, affine), image_path)
            nib.save(nib.Nifti1Image(target, affine), label_path)
            nib.save(nib.Nifti1Image(proposal, affine), proposals / "case.nii.gz")
            result = process_case(
                {
                    "item": {
                        "name": "case",
                        "image": str(image_path),
                        "label": str(label_path),
                        "cohort": "partial",
                        "prediction_name": "case.nii.gz",
                        "complete_label": False,
                        "positive_only_label": True,
                        "training_only": True,
                        "label_contract": "partial_primary_positive_only",
                    },
                    "proposal_dirs": [str(proposals)],
                    "source_names": ["flare23"],
                    "output": str(output),
                    "shape": (16, 16, 16),
                    "crop_spacing": (1.0, 1.0, 1.0),
                    "min_mm3": 1.0,
                    "max_mm3": 1000.0,
                    "max_negatives": 8,
                    "min_negatives_per_source": 0,
                    "bbox_margin_mm": 0.0,
                    "tile_stride_mm": 0.0,
                    "max_negative_tiles": 1,
                    "component_policy": "source_separate",
                    "compact_source_components": True,
                    "resume": False,
                }
            )
            self.assertTrue(result["rows"])
            self.assertTrue(all(int(row["label"]) == 1 for row in result["rows"]))
            self.assertTrue(
                all(int(row["training_only"]) == 1 for row in result["rows"])
            )
            self.assertTrue(
                all(
                    row["label_contract"] == "partial_primary_positive_only"
                    for row in result["rows"]
                )
            )

    def test_resume_reuses_only_complete_matching_case(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proposals = root / "proposals"
            output = root / "output"
            proposals.mkdir()
            affine = np.eye(4)
            ct = np.full((24, 24, 24), 30.0, dtype=np.float32)
            target = np.zeros_like(ct, dtype=np.uint8)
            target[5:9, 5:9, 5:9] = 1
            proposal = np.zeros_like(target)
            proposal[6:8, 6:8, 6:8] = 1
            proposal[16:19, 16:19, 16:19] = 1
            image_path = root / "case_0000.nii.gz"
            label_path = root / "case.nii.gz"
            nib.save(nib.Nifti1Image(ct, affine), image_path)
            nib.save(nib.Nifti1Image(target, affine), label_path)
            nib.save(nib.Nifti1Image(proposal, affine), proposals / "case.nii.gz")
            task = {
                "item": {
                    "name": "case",
                    "image": str(image_path),
                    "label": str(label_path),
                    "cohort": "synthetic",
                    "prediction_name": "case.nii.gz",
                    "complete_label": True,
                },
                "proposal_dirs": [str(proposals)],
                "source_names": ["source"],
                "output": str(output),
                "shape": (16, 16, 16),
                "crop_spacing": (1.0, 1.0, 1.0),
                "min_mm3": 1.0,
                "max_mm3": 1000.0,
                "max_negatives": 8,
                "min_negatives_per_source": 0,
                "bbox_margin_mm": 0.0,
                "tile_stride_mm": 0.0,
                "max_negative_tiles": 1,
                "component_policy": "source_separate",
                "compact_source_components": True,
                "resume": True,
            }

            first = process_case(task)
            sidecars = list((output / "case_rows").glob("*.json"))
            self.assertEqual(len(sidecars), 1)
            with patch(
                "prepare_candidate_refiner_patches.nib.load",
                side_effect=AssertionError("resume unexpectedly loaded imaging"),
            ):
                resumed = process_case(task)
            self.assertTrue(resumed["resumed"])
            self.assertEqual(resumed["rows"], first["rows"])

            # A recipe change must never reuse rows produced by another recipe.
            changed = {**task, "min_mm3": 2.0}
            self.assertIsNone(
                load_resumable_case(changed, output, "case", "synthetic")
            )

            # A missing patch invalidates the case sidecar and triggers a full,
            # deterministic recomputation rather than silently yielding a hole.
            missing = Path(first["rows"][0]["path"])
            missing.unlink()
            recomputed = process_case(task)
            self.assertNotIn("resumed", recomputed)
            self.assertTrue(missing.is_file())
            self.assertEqual(recomputed["rows"], first["rows"])

    def test_compact_source_records_are_exactly_legacy_equivalent(self):
        shape = (24, 22, 20)
        first = np.zeros(shape, dtype=bool)
        second = np.zeros(shape, dtype=bool)
        third = np.zeros(shape, dtype=bool)
        first[2:6, 3:8, 4:9] = True
        first[14:18, 12:16, 10:14] = True
        second[2:6, 3:8, 4:9] = True  # exact duplicate
        second[6:9, 3:8, 4:9] = True  # touching, distinct source component
        third[15:17, 13:15, 11:13] = True  # nested, not identical
        target = np.zeros(shape, dtype=bool)
        target[3:5, 4:6, 5:7] = True
        target[15:17, 13:15, 11:13] = True
        masks = [first, second, third]
        support = np.sum(masks, axis=0, dtype=np.uint8)

        legacy, component_maps, legacy_target = source_separated_component_records(
            masks, support, target, np.asarray((1.2, 1.4, 1.6)), 1.0, 1e9
        )
        compact, compact_target = source_separated_component_records_compact(
            masks, support, target, np.asarray((1.2, 1.4, 1.6)), 1.0, 1e9
        )
        np.testing.assert_array_equal(compact_target, legacy_target)
        self.assertEqual(len(compact), len(legacy))
        private = {"_component_shape", "_component_bits"}
        for expected, observed in zip(legacy, compact):
            self.assertEqual(set(expected), set(observed) - private)
            for key, value in expected.items():
                if isinstance(value, np.ndarray):
                    np.testing.assert_allclose(observed[key], value, rtol=0, atol=0)
                else:
                    self.assertEqual(observed[key], value)
            bbox = tuple(expected["bbox"])
            local = (
                component_maps[int(expected["source_index"])][bbox]
                == int(expected["source_component_id"])
            )
            np.testing.assert_array_equal(unpack_local_component(observed), local)
        self.assertEqual(
            [row["component_id"] for row in deterministic_negative_subset(
                compact, "case", maximum=2, minimum_per_source=0
            )],
            [row["component_id"] for row in deterministic_negative_subset(
                legacy, "case", maximum=2, minimum_per_source=0
            )],
        )

    def test_component_targets_and_hard_negative_selection(self):
        proposal = np.zeros((20, 20, 20), dtype=bool)
        proposal[4:7, 4:7, 4:7] = True
        proposal[10:13, 10:13, 10:13] = True
        proposal[16:18, 16:18, 16:18] = True
        target = np.zeros_like(proposal)
        target[5:8, 5:8, 5:8] = True
        support = proposal.astype(np.uint8)
        records, component_map, target_map = component_records(
            proposal, support, target, np.ones(3), 1.0, 1000.0
        )
        self.assertEqual(component_map.shape, proposal.shape)
        self.assertEqual(target_map.shape, target.shape)
        self.assertEqual(sum(record["positive"] for record in records), 1)
        kept = deterministic_negative_subset(records, "case", maximum=1)
        self.assertEqual(len(kept), 2)
        negative = [record for record in kept if not record["positive"]][0]
        self.assertLess(negative["distance_to_target_mm"], 10.0)

    def test_source_minimum_prevents_dense_source_from_crowding_out_others(self):
        records = []
        component = 1
        for source, count in ((0, 20), (1, 2), (2, 2)):
            for index in range(count):
                records.append(
                    {
                        "component_id": component,
                        "source_index": source,
                        "positive": False,
                        "distance_to_target_mm": float(index + source),
                        "volume_mm3": float(100 - index),
                    }
                )
                component += 1
        kept = deterministic_negative_subset(
            records, "case", maximum=6, minimum_per_source=1
        )
        counts = {
            source: sum(int(row["source_index"]) == source for row in kept)
            for source in (0, 1, 2)
        }
        self.assertEqual(len(kept), 6)
        self.assertTrue(all(value >= 1 for value in counts.values()))

    def test_physical_crop_is_centered_and_preserves_binary_mask(self):
        source = np.zeros((24, 24, 24), dtype=np.uint8)
        source[11:14, 11:14, 11:14] = 1
        source_affine = np.diag((2.0, 2.0, 2.0, 1.0))
        center_world = nib.affines.apply_affine(source_affine, (12, 12, 12))
        target_affine = crop_affine(center_world, (16, 16, 16), (2.0, 2.0, 2.0))
        crop = resample_array(
            source, source_affine, target_affine, (16, 16, 16), 0, 0.0
        )
        self.assertEqual(set(np.unique(crop)), {0, 1})
        centroid = np.argwhere(crop > 0).mean(axis=0)
        np.testing.assert_allclose(centroid, (7.5, 7.5, 7.5), atol=1.0)

    def test_process_case_writes_positive_and_negative_patches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proposals = root / "proposals"
            output = root / "output"
            proposals.mkdir()
            affine = np.eye(4)
            ct = np.full((24, 24, 24), 30.0, dtype=np.float32)
            target = np.zeros_like(ct, dtype=np.uint8)
            target[5:9, 5:9, 5:9] = 1
            proposal = np.zeros_like(target)
            proposal[6:8, 6:8, 6:8] = 1
            proposal[16:19, 16:19, 16:19] = 1
            image_path = root / "case_0000.nii.gz"
            label_path = root / "case.nii.gz"
            proposal_path = proposals / "case.nii.gz"
            nib.save(nib.Nifti1Image(ct, affine), image_path)
            nib.save(nib.Nifti1Image(target, affine), label_path)
            nib.save(nib.Nifti1Image(proposal, affine), proposal_path)
            result = process_case(
                {
                    "item": {
                        "name": "case",
                        "image": str(image_path),
                        "label": str(label_path),
                        "cohort": "synthetic",
                        "prediction_name": "case.nii.gz",
                        "complete_label": True,
                    },
                    "proposal_dirs": [str(proposals)],
                    "source_names": ["source"],
                    "output": str(output),
                    "shape": (16, 16, 16),
                    "crop_spacing": (1.0, 1.0, 1.0),
                    "min_mm3": 1.0,
                    "max_mm3": 1000.0,
                    "max_negatives": 8,
                }
            )
            self.assertEqual(len(result["rows"]), 2)
            self.assertEqual(sorted(row["label"] for row in result["rows"]), [0, 1])
            for row in result["rows"]:
                with np.load(row["path"], allow_pickle=False) as archive:
                    self.assertEqual(archive["image"].shape, (3, 16, 16, 16))
                    self.assertEqual(archive["target"].shape, (1, 16, 16, 16))
                audited = audit_row((row, (16, 16, 16)))
                self.assertEqual(audited["label"], int(row["label"]))
                self.assertGreater(audited["seed_voxels"], 0)

    def test_source_separate_policy_preserves_touching_source_components(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proposal_a = root / "proposal_a"
            proposal_b = root / "proposal_b"
            output = root / "output"
            proposal_a.mkdir()
            proposal_b.mkdir()
            affine = np.eye(4)
            ct = np.zeros((20, 20, 20), dtype=np.float32)
            target = np.zeros_like(ct, dtype=np.uint8)
            target[3:5, 4:6, 4:6] = 1
            first = np.zeros_like(target)
            second = np.zeros_like(target)
            first[3:5, 4:6, 4:6] = 1
            second[5:7, 4:6, 4:6] = 1
            image_path = root / "case_0000.nii.gz"
            label_path = root / "case.nii.gz"
            nib.save(nib.Nifti1Image(ct, affine), image_path)
            nib.save(nib.Nifti1Image(target, affine), label_path)
            nib.save(nib.Nifti1Image(first, affine), proposal_a / "case.nii.gz")
            nib.save(nib.Nifti1Image(second, affine), proposal_b / "case.nii.gz")

            result = process_case(
                {
                    "item": {
                        "name": "case",
                        "image": str(image_path),
                        "label": str(label_path),
                        "cohort": "synthetic",
                        "prediction_name": "case.nii.gz",
                        "complete_label": True,
                    },
                    "proposal_dirs": [str(proposal_a), str(proposal_b)],
                    "source_names": ["a", "b"],
                    "output": str(output),
                    "shape": (12, 12, 12),
                    "crop_spacing": (1.0, 1.0, 1.0),
                    "min_mm3": 1.0,
                    "max_mm3": 1000.0,
                    "max_negatives": 8,
                    "component_policy": "source_separate",
                }
            )

            self.assertEqual(len(result["rows"]), 2)
            self.assertEqual(
                [row["patch_id"] for row in result["rows"]],
                ["case__c0001", "case__c0002"],
            )
            self.assertEqual(
                [row["source_name"] for row in result["rows"]], ["a", "b"]
            )
            self.assertEqual(
                [row["candidate_positive"] for row in result["rows"]], [1, 0]
            )

    def test_compact_process_case_matches_legacy_rows_and_archives(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proposal_a = root / "proposal_a"
            proposal_b = root / "proposal_b"
            proposal_c = root / "proposal_c"
            for directory in (proposal_a, proposal_b, proposal_c):
                directory.mkdir()
            affine = np.diag((1.2, 1.4, 1.6, 1.0))
            shape = (30, 28, 26)
            ct = np.linspace(-900, 300, np.prod(shape), dtype=np.float32).reshape(shape)
            target = np.zeros(shape, dtype=np.uint8)
            target[4:7, 6:9, 8:11] = 1
            target[18:20, 17:20, 14:17] = 1
            first = np.zeros(shape, dtype=np.uint8)
            second = np.zeros(shape, dtype=np.uint8)
            third = np.zeros(shape, dtype=np.uint8)
            first[3:12, 5:10, 7:12] = 1
            first[22:25, 21:24, 20:23] = 1
            second[3:12, 5:10, 7:12] = 1  # exact duplicate
            second[12:16, 5:10, 7:12] = 1  # touching alternative
            third[17:22, 16:21, 13:18] = 1
            image_path = root / "case_0000.nii.gz"
            label_path = root / "case.nii.gz"
            nib.save(nib.Nifti1Image(ct, affine), image_path)
            nib.save(nib.Nifti1Image(target, affine), label_path)
            for directory, mask in zip(
                (proposal_a, proposal_b, proposal_c), (first, second, third)
            ):
                nib.save(nib.Nifti1Image(mask, affine), directory / "case.nii.gz")

            common = {
                "item": {
                    "name": "case",
                    "image": str(image_path),
                    "label": str(label_path),
                    "cohort": "synthetic",
                    "prediction_name": "case.nii.gz",
                    "complete_label": True,
                },
                "proposal_dirs": [str(proposal_a), str(proposal_b), str(proposal_c)],
                "source_names": ["a", "b", "c"],
                "shape": (16, 16, 16),
                "crop_spacing": (1.5, 1.5, 1.5),
                "min_mm3": 1.0,
                "max_mm3": 1e9,
                "max_negatives": 8,
                "min_negatives_per_source": 1,
                "bbox_margin_mm": 4.0,
                "tile_stride_mm": 12.0,
                "max_negative_tiles": 1,
                "component_policy": "source_separate",
            }
            legacy = process_case(
                {**common, "output": str(root / "legacy")}
            )
            compact = process_case(
                {
                    **common,
                    "output": str(root / "compact"),
                    "compact_source_components": True,
                }
            )
            self.assertEqual(len(compact["rows"]), len(legacy["rows"]))
            for expected, observed in zip(legacy["rows"], compact["rows"]):
                self.assertEqual(
                    {key: value for key, value in observed.items() if key != "path"},
                    {key: value for key, value in expected.items() if key != "path"},
                )
                with np.load(expected["path"], allow_pickle=False) as old, np.load(
                    observed["path"], allow_pickle=False
                ) as new:
                    self.assertEqual(set(old.files), set(new.files))
                    for key in old.files:
                        np.testing.assert_array_equal(new[key], old[key])

    def test_seed_prompt_survives_subvoxel_resampling(self):
        source = np.zeros((8, 8, 8), dtype=np.uint8)
        source[3, 3, 3] = 1
        source_affine = np.diag((3.0, 3.0, 3.0, 1.0))
        center = nib.affines.apply_affine(source_affine, (3, 3, 3))
        target_affine = crop_affine(center, (16, 16, 16), (1.5, 1.5, 1.5))
        seed, fallback = resample_seed_prompt(
            source, source_affine, target_affine, (16, 16, 16), 1
        )
        self.assertGreater(float(seed.max()), 0)
        self.assertFalse(fallback)

    def test_bbox_local_seed_resampling_matches_full_volume(self):
        source = np.zeros((30, 28, 26), dtype=np.float32)
        source[9:15, 7:13, 11:18] = 1.0
        source[11:13, 9:11, 13:16] = 2.0
        source_affine = np.diag((1.2, 1.7, 2.1, 1.0))
        tight_bbox = (slice(9, 15), slice(7, 13), slice(11, 18))
        bbox = pad_slices(tight_bbox, source.shape)
        center_world = nib.affines.apply_affine(source_affine, (12, 10, 14))
        target_affine = crop_affine(
            center_world, (20, 20, 20), (1.5, 1.5, 1.5)
        )
        full_seed, full_fallback = resample_seed_prompt(
            source, source_affine, target_affine, (20, 20, 20), 2
        )
        local_seed, local_fallback = resample_seed_prompt(
            source[bbox],
            slice_affine(source_affine, bbox),
            target_affine,
            (20, 20, 20),
            2,
        )
        self.assertEqual(full_fallback, local_fallback)
        np.testing.assert_allclose(local_seed, full_seed, atol=1e-6, rtol=0)

    def test_bbox_tiling_uses_tile_level_targets_and_caps_negative_tiles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proposals = root / "proposals"
            output = root / "output"
            proposals.mkdir()
            affine = np.eye(4)
            ct = np.full((40, 40, 40), 30.0, dtype=np.float32)
            target = np.zeros_like(ct, dtype=np.uint8)
            target[4:6, 10:12, 10:12] = 1
            proposal = np.zeros_like(target)
            proposal[4:30, 10:12, 10:12] = 1
            proposal[4:30, 30:32, 30:32] = 1
            image_path = root / "case_0000.nii.gz"
            label_path = root / "case.nii.gz"
            nib.save(nib.Nifti1Image(ct, affine), image_path)
            nib.save(nib.Nifti1Image(target, affine), label_path)
            nib.save(nib.Nifti1Image(proposal, affine), proposals / "case.nii.gz")

            result = process_case(
                {
                    "item": {
                        "name": "case",
                        "image": str(image_path),
                        "label": str(label_path),
                        "cohort": "synthetic",
                        "prediction_name": "case.nii.gz",
                        "complete_label": True,
                    },
                    "proposal_dirs": [str(proposals)],
                    "source_names": ["source"],
                    "output": str(output),
                    "shape": (16, 16, 16),
                    "crop_spacing": (1.0, 1.0, 1.0),
                    "min_mm3": 1.0,
                    "max_mm3": 1000.0,
                    "max_negatives": 8,
                    "bbox_margin_mm": 4.0,
                    "tile_stride_mm": 12.0,
                    "max_negative_tiles": 1,
                }
            )

            positive_candidate = [
                row for row in result["rows"] if row["candidate_positive"] == 1
            ]
            negative_candidate = [
                row for row in result["rows"] if row["candidate_positive"] == 0
            ]
            self.assertEqual(len(positive_candidate), 3)
            self.assertEqual(sum(row["label"] for row in positive_candidate), 1)
            self.assertEqual(len(negative_candidate), 1)
            self.assertTrue(all("__t" in row["patch_id"] for row in result["rows"]))

    def test_empty_resampled_seed_gets_deterministic_center_prompt(self):
        source = np.zeros((8, 8, 8), dtype=np.uint8)
        affine = np.eye(4)
        seed, fallback = resample_seed_prompt(
            source, affine, affine, (16, 16, 16), 1
        )
        self.assertTrue(fallback)
        self.assertEqual(int(np.count_nonzero(seed)), 8)
        self.assertTrue(np.all(seed[7:9, 7:9, 7:9] == 1))

    def test_subvoxel_positive_target_is_preserved_at_physical_location(self):
        source = np.zeros((8, 8, 8), dtype=np.uint8)
        source[3, 3, 3] = 1
        source_affine = np.diag((0.5, 0.5, 1.0, 1.0))
        center_world = nib.affines.apply_affine(source_affine, (3, 3, 3))
        target_affine = crop_affine(
            center_world, (16, 16, 16), (1.5, 1.5, 1.5)
        )
        target, fallback = resample_binary_target(
            source, source_affine, target_affine, (16, 16, 16)
        )
        self.assertTrue(fallback)
        self.assertEqual(int(target.sum()), 1)


if __name__ == "__main__":
    unittest.main()
