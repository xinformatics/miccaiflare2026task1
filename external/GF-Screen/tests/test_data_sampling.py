import unittest
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path
from unittest import mock

import torch

from utils.data_utils import (
    ComponentUniformFgBgIndicesd,
    EmptyCropTrustDataset,
    Sampler,
    _cohort_key,
    _source_dataset_prefix,
    build_cohort_sampling_weights,
    component_candidate_counts,
    load_cohort_priority_weights,
    load_complete_label_cohorts,
)


class ComponentUniformCropTests(unittest.TestCase):
    def test_positive_only_cache_identity_remains_v2_compatible(self):
        transform = ComponentUniformFgBgIndicesd(
            points_per_component=64,
            size_power=0.5,
            background_points=0,
            cache_dir="/tmp/component-cache-test",
        )
        name = "legacy-case"
        shape = (1, 8, 8, 8)
        identity = (
            "component-uniform-v2-cc3d-boxes|legacy-case|"
            "(1, 8, 8, 8)|64|0.500000"
        )
        expected = hashlib.sha256(identity.encode("utf-8")).hexdigest() + ".pt"
        self.assertEqual(transform._cache_path(name, shape).name, expected)
        with_background = ComponentUniformFgBgIndicesd(
            points_per_component=64,
            size_power=0.5,
            background_points=4096,
            cache_dir="/tmp/component-cache-test",
        )
        self.assertNotEqual(
            transform._cache_path(name, shape),
            with_background._cache_path(name, shape),
        )

    def test_zero_power_is_exactly_component_uniform(self):
        counts = component_candidate_counts([1, 100, 10000], 16, 0.0)
        self.assertEqual(counts.tolist(), [16, 16, 16])

    def test_square_root_balance_is_bounded_between_uniform_and_voxel(self):
        counts = component_candidate_counts([1, 100, 10000], 100, 0.5)
        self.assertEqual(int(counts.sum()), 300)
        self.assertTrue(torch.all(torch.as_tensor(counts) >= 1))
        self.assertGreater(counts[2], counts[1])
        self.assertGreater(counts[1], counts[0])
        self.assertLess(counts[2] / counts[0], 10000)

    def test_every_component_has_equal_candidate_mass(self):
        label = torch.zeros((1, 12, 12, 12), dtype=torch.float32)
        image = torch.ones_like(label)
        label[0, 1, 1, 1] = 1
        label[0, 6:10, 6:10, 6:10] = 1
        transform = ComponentUniformFgBgIndicesd(points_per_component=16)
        result = transform(
            {"image": image, "label": label, "name": "unequal-components"}
        )
        indices = result["component_fg_indices"]
        coordinates = torch.stack(torch.unravel_index(indices, label.shape[1:]), dim=1)
        small = torch.all(coordinates == torch.tensor([1, 1, 1]), dim=1)
        self.assertEqual(len(indices), 32)
        self.assertEqual(int(small.sum()), 16)
        self.assertTrue(torch.all(label[0].reshape(-1)[indices] > 0))

    def test_diagonal_voxels_use_26_connectivity(self):
        label = torch.zeros((1, 4, 4, 4), dtype=torch.float32)
        image = torch.ones_like(label)
        label[0, 1, 1, 1] = 1
        label[0, 2, 2, 2] = 1
        transform = ComponentUniformFgBgIndicesd(points_per_component=8)
        result = transform({"image": image, "label": label, "name": "diagonal"})
        self.assertEqual(len(result["component_fg_indices"]), 8)

    def test_bounded_background_candidates_are_inside_body_and_outside_label(self):
        label = torch.zeros((1, 8, 8, 8), dtype=torch.float32)
        image = torch.zeros_like(label)
        image[0, 2:7, 2:7, 2:7] = 0.5
        label[0, 3:5, 3:5, 3:5] = 1
        transform = ComponentUniformFgBgIndicesd(
            points_per_component=8, background_points=32
        )
        result = transform(
            {"image": image, "label": label, "name": "bounded-background"}
        )
        indices = result["component_bg_indices"]
        self.assertEqual(len(indices), 32)
        self.assertTrue(torch.all(label[0].reshape(-1)[indices] == 0))
        self.assertTrue(torch.all(image[0].reshape(-1)[indices] > 0))
        replay = transform(
            {"image": image, "label": label, "name": "bounded-background"}
        )
        self.assertTrue(torch.equal(indices, replay["component_bg_indices"]))

    def test_background_mode_reuses_separate_legacy_positive_cache(self):
        label = torch.zeros((1, 8, 8, 8), dtype=torch.float32)
        image = torch.ones_like(label)
        label[0, 2:4, 2:4, 2:4] = 1
        with tempfile.TemporaryDirectory() as legacy_directory, tempfile.TemporaryDirectory() as new_directory:
            legacy_transform = ComponentUniformFgBgIndicesd(
                points_per_component=7,
                size_power=0.5,
                cache_dir=legacy_directory,
            )
            legacy = legacy_transform(
                {"image": image, "label": label, "name": "legacy-reuse"}
            )
            transform = ComponentUniformFgBgIndicesd(
                points_per_component=7,
                size_power=0.5,
                background_points=16,
                cache_dir=new_directory,
                legacy_cache_dir=legacy_directory,
            )
            with mock.patch.object(
                transform, "_compute", side_effect=AssertionError("recomputed components")
            ):
                result = transform(
                    {"image": image, "label": label, "name": "legacy-reuse"}
                )
            self.assertTrue(
                torch.equal(result["component_fg_indices"], legacy["component_fg_indices"])
            )
            self.assertEqual(len(result["component_bg_indices"]), 16)

    def test_sidecar_is_reproducible_and_small(self):
        label = torch.zeros((1, 8, 8, 8), dtype=torch.float32)
        image = torch.ones_like(label)
        label[0, 1:3, 1:3, 1:3] = 1
        label[0, 5:7, 5:7, 5:7] = 1
        with tempfile.TemporaryDirectory() as directory:
            transform = ComponentUniformFgBgIndicesd(
                points_per_component=7, cache_dir=directory
            )
            first = transform({"image": image, "label": label, "name": "cached-case"})
            second = transform({"image": image, "label": label, "name": "cached-case"})
            self.assertTrue(
                torch.equal(first["component_fg_indices"], second["component_fg_indices"])
            )
            files = list(Path(directory).glob("*.pt"))
            self.assertEqual(len(files), 1)
            self.assertLess(files[0].stat().st_size, 10000)
            locks = list(Path(directory).glob("*.pt.lock"))
            self.assertEqual(len(locks), 1)

    def test_shared_cache_rechecks_under_lock_before_computing(self):
        label = torch.zeros((1, 8, 8, 8), dtype=torch.float32)
        image = torch.ones_like(label)
        label[0, 2:4, 2:4, 2:4] = 1
        with tempfile.TemporaryDirectory() as directory:
            first = ComponentUniformFgBgIndicesd(
                points_per_component=7, cache_dir=directory
            )
            expected = first(
                {"image": image, "label": label, "name": "shared-case"}
            )
            second = ComponentUniformFgBgIndicesd(
                points_per_component=7, cache_dir=directory
            )
            with mock.patch.object(
                second, "_build_payload", side_effect=AssertionError("duplicate build")
            ):
                observed = second(
                    {"image": image, "label": label, "name": "shared-case"}
                )
            self.assertTrue(
                torch.equal(expected["component_fg_indices"], observed["component_fg_indices"])
            )
            self.assertTrue(
                torch.equal(expected["component_bg_indices"], observed["component_bg_indices"])
            )


class SizedDataset:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


class ItemDataset:
    def __init__(self, names):
        self.data = [{"name": name} for name in names]


class SourceDatasetPrefixTests(unittest.TestCase):
    def test_complete_label_manifest_and_wrapper_are_explicit(self):
        payload = {
            "complete_label_cohorts": [
                "abdomen:WholeBody_autoPETCT",
                "chest:Chest_LNDb",
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "complete.json"
            path.write_text(json.dumps(payload))
            cohorts = load_complete_label_cohorts(path)
        # Special methods are looked up on the class, so use a minimal local
        # dataset with the same public interface for wrapper behavior.
        class GettableItemDataset(ItemDataset):
            def __getitem__(self, index):
                return self.data[index]

        wrapped = EmptyCropTrustDataset(
            GettableItemDataset([
                "WholeBody_autoPETCT_0001",
                "WholeBody_DeepLesion_0002",
            ]),
            trusted_empty=False,
            region="abdomen",
            complete_label_cohorts=cohorts,
        )
        self.assertTrue(bool(wrapped[0]["complete_label"]))
        self.assertFalse(bool(wrapped[1]["complete_label"]))
        self.assertFalse(bool(wrapped[0]["trusted_empty"]))

    def test_complete_label_manifest_rejects_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "complete.json"
            path.write_text(json.dumps({"complete_label_cohorts": ["a", "a"]}))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_complete_label_cohorts(path)

    def test_challenge_name_conventions(self):
        expected = {
            "WholeBody_DeepLesion_000271-09-01": "WholeBody_DeepLesion",
            "WholeBody_autoPETCT_0419": "WholeBody_autoPETCT",
            "Chest_LIDC-IDRI_0001": "Chest_LIDC-IDRI",
            "Pancreas_PanTrack_010-20230328": "Pancreas_PanTrack",
            "Liver_WAW-TACE_Arterial_001": "Liver_WAW-TACE",
            "Liver-MSD-HepaticVessel_078": "Liver-MSD-HepaticVessel",
            "Adrenal-ACC-Ki67-Seg_001": "Adrenal-ACC-Ki67-Seg",
        }
        for name, source in expected.items():
            with self.subTest(name=name):
                self.assertEqual(_source_dataset_prefix(name), source)

    def test_real_training_jsons_resolve_28_sources(self):
        root = Path(__file__).resolve().parents[1]
        counts = Counter()
        for region, relative in (
            ("abdomen", "jsons/local/abdomen.json"),
            ("chest", "jsons/local/chest.json"),
        ):
            with (root / relative).open() as stream:
                rows = json.load(stream)["training"]
            counts.update(_cohort_key(row, region) for row in rows)
        self.assertEqual(len(counts), 28)
        self.assertEqual(counts["abdomen:WholeBody_DeepLesion"], 4978)
        self.assertEqual(counts["abdomen:WholeBody_autoPETCT"], 687)
        self.assertEqual(counts["chest:Chest_LUNA25"], 4017)
        self.assertEqual(counts["chest:Chest_LIDC-IDRI"], 868)

    def test_balancing_separates_collapsed_sources_and_preserves_mass(self):
        dataset = ItemDataset([
            "WholeBody_DeepLesion_1",
            "WholeBody_DeepLesion_2",
            "WholeBody_MSWAL_1",
        ])
        weights, counts, _ = build_cohort_sampling_weights(
            [("abdomen", dataset)], power=0.5, max_weight=5.0
        )
        self.assertEqual(counts["abdomen:WholeBody_DeepLesion"], 2)
        self.assertEqual(counts["abdomen:WholeBody_MSWAL"], 1)
        self.assertTrue(torch.isfinite(weights).all())
        self.assertAlmostEqual(float(weights.sum()), 3.0, places=12)
        self.assertGreater(float(weights[2]), float(weights[0]))

    def test_priority_weights_preserve_mass_and_raise_selected_source(self):
        dataset = ItemDataset([
            "WholeBody_DeepLesion_1",
            "WholeBody_DeepLesion_2",
            "WholeBody_MSWAL_1",
            "WholeBody_MSWAL_2",
        ])
        weights, _, cohort_weights = build_cohort_sampling_weights(
            [("abdomen", dataset)],
            priority_weights={"abdomen:WholeBody_MSWAL": 2.0},
        )
        self.assertAlmostEqual(float(weights.sum()), 4.0, places=12)
        self.assertGreater(
            cohort_weights["abdomen:WholeBody_MSWAL"],
            cohort_weights["abdomen:WholeBody_DeepLesion"],
        )

    def test_versioned_multilesion_priority_file_matches_real_sources(self):
        root = Path(__file__).resolve().parents[1]
        priorities = load_cohort_priority_weights(
            root / "configs/multilesion_source_priority_20260820.json"
        )
        self.assertEqual(priorities["abdomen:WholeBody_autoPETCT"], 2.0)
        self.assertEqual(priorities["chest:Chest_LNDb"], 2.0)

    def test_real_multilesion_priority_exposure_and_mass(self):
        root = Path(__file__).resolve().parents[1]
        priorities = load_cohort_priority_weights(
            root / "configs/multilesion_source_priority_20260820.json"
        )
        groups = []
        keys = []
        for region, relative in (
            ("abdomen", "jsons/local/abdomen.json"),
            ("chest", "jsons/local/chest.json"),
        ):
            with (root / relative).open() as stream:
                rows = json.load(stream)["training"]
            groups.append((region, ItemDataset([row["name"] for row in rows])))
            keys.extend(_cohort_key(row, region) for row in rows)
        weights, _, _ = build_cohort_sampling_weights(
            groups, priority_weights=priorities
        )
        selected = torch.tensor([key in priorities for key in keys])
        self.assertEqual(len(weights), 17445)
        self.assertAlmostEqual(float(weights.sum()), 17445.0, places=8)
        self.assertEqual(int(selected.sum()), 3451)
        self.assertAlmostEqual(
            float(weights[selected].sum() / weights.sum()), 0.33030245, places=7
        )


class EpochSamplerTests(unittest.TestCase):
    def test_two_rank_sampler_shards_one_global_epoch(self):
        samplers = [
            Sampler(
                SizedDataset(20000),
                num_replicas=2,
                rank=rank,
                num_samples_total=8192,
            )
            for rank in range(2)
        ]
        for sampler in samplers:
            sampler.set_epoch(283)
        rank_indices = [list(sampler) for sampler in samplers]
        combined = rank_indices[0] + rank_indices[1]
        # The requested 8,192-case global epoch is split evenly; ranks do not
        # each draw an independent 8,192-case epoch.
        self.assertEqual([len(indices) for indices in rank_indices], [4096, 4096])
        self.assertEqual(len(combined), 8192)
        self.assertEqual(len(set(combined)), 8192)
        self.assertTrue(set(rank_indices[0]).isdisjoint(rank_indices[1]))

    def test_unweighted_epoch_replay(self):
        sampler = Sampler(
            SizedDataset(10), num_replicas=1, rank=0, num_samples_total=5
        )
        sampler.set_epoch(7)
        first = list(sampler)
        sampler.set_epoch(7)
        replay = list(sampler)
        sampler.set_epoch(8)
        next_epoch = list(sampler)
        self.assertEqual(first, replay)
        self.assertNotEqual(first, next_epoch)
        self.assertEqual(len(first), 5)
        self.assertEqual(len(set(first)), 5)

    def test_weighted_epoch_replay(self):
        weights = torch.tensor([1.0] * 5 + [10.0] * 5)
        sampler = Sampler(
            SizedDataset(10),
            num_replicas=1,
            rank=0,
            num_samples_total=5,
            weights=weights,
        )
        sampler.set_epoch(7)
        first = list(sampler)
        sampler.set_epoch(7)
        replay = list(sampler)
        sampler.set_epoch(8)
        next_epoch = list(sampler)
        self.assertEqual(first, replay)
        self.assertNotEqual(first, next_epoch)
        self.assertEqual(len(first), 5)


if __name__ == "__main__":
    unittest.main()
