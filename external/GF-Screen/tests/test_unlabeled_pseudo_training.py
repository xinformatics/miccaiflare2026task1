import json
import hashlib
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.data import list_data_collate

from build_unlabeled_hysteresis_pseudolabels import (
    build_case,
    build_records,
    discover_images,
)
from utils.data_utils import EmptyCropTrustDataset, load_positive_pseudo_datalist


def save_mask(path, array, spacing=(1.0, 1.0, 1.0)):
    affine = np.diag([*spacing, 1.0])
    nib.save(nib.Nifti1Image(array.astype(np.uint8), affine), path)


class HysteresisPseudoLabelTests(unittest.TestCase):
    def test_retains_only_support_component_with_large_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            support = np.zeros((12, 12, 12), dtype=np.uint8)
            support[1:5, 1:5, 1:5] = 1
            support[8:10, 8:10, 8:10] = 1
            seed = np.zeros_like(support)
            seed[2:4, 2:4, 2:4] = 1
            seed[8, 8, 8] = 1
            save_mask(root / "support.nii.gz", support)
            save_mask(root / "seed.nii.gz", seed)

            stats = build_case(
                root / "support.nii.gz",
                root / "seed.nii.gz",
                root / "output.nii.gz",
                min_seed_mm3=4,
                min_component_mm3=10,
            )
            output = np.asarray(nib.load(root / "output.nii.gz").dataobj)
            self.assertEqual(stats["components"], 1)
            self.assertEqual(int(output.sum()), 64)
            self.assertEqual(int(output[8:10, 8:10, 8:10].sum()), 0)

    def test_rejects_non_nested_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            support = np.zeros((5, 5, 5), dtype=np.uint8)
            seed = np.zeros_like(support)
            seed[2, 2, 2] = 1
            save_mask(root / "support.nii.gz", support)
            save_mask(root / "seed.nii.gz", seed)
            with self.assertRaisesRegex(ValueError, "not nested"):
                build_case(
                    root / "support.nii.gz",
                    root / "seed.nii.gz",
                    root / "output.nii.gz",
                    min_seed_mm3=1,
                    min_component_mm3=1,
                )

    def test_recursive_images_require_unique_prediction_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").mkdir()
            (root / "b").mkdir()
            save_mask(root / "a" / "case_0000.nii.gz", np.zeros((2, 2, 2)))
            save_mask(root / "b" / "case.nii.gz", np.zeros((2, 2, 2)))
            with self.assertRaisesRegex(ValueError, "duplicate prediction basename"):
                discover_images(root)

    def test_parallel_records_preserve_sorted_task_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = []
            for index in range(2):
                support = np.zeros((8, 8, 8), dtype=np.uint8)
                seed = np.zeros_like(support)
                support[1:5, 1:5, 1:5] = 1
                seed[2:4, 2:4, 2:4] = 1
                image = root / f"image_{index}.nii.gz"
                support_path = root / f"support_{index}.nii.gz"
                seed_path = root / f"seed_{index}.nii.gz"
                save_mask(image, np.zeros_like(support))
                save_mask(support_path, support)
                save_mask(seed_path, seed)
                tasks.append((
                    f"case_{index}.nii.gz", image, support_path, seed_path,
                    root / f"output_{index}.nii.gz", 4, 10, "test",
                ))
            records = build_records(tasks, workers=2)
            self.assertEqual(
                [case["name"] for case, _ in records], ["case_0", "case_1"]
            )
            self.assertTrue(all(item["pseudo_label"] for _, item in records))
            for case, item in records:
                label = Path(case["label"])
                digest = hashlib.sha256(label.read_bytes()).hexdigest()
                self.assertEqual(case["label_sha256"], digest)
                self.assertEqual(item["label_sha256"], digest)
                self.assertEqual(case["label_bytes"], label.stat().st_size)
                self.assertEqual(item["label_bytes"], label.stat().st_size)


class PositivePseudoManifestTests(unittest.TestCase):
    def test_requires_explicit_pseudo_provenance_and_repeats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "training": [
                            {
                                "image": "image.nii.gz",
                                "label": "label.nii.gz",
                                "label_sha256": "immutable-audit-digest",
                                "label_bytes": 123,
                                "name": "pseudo_case",
                                "pseudo_label": True,
                            }
                        ],
                        "validation": [],
                    }
                )
            )
            items = load_positive_pseudo_datalist(manifest, root, repeat=3)
            self.assertEqual(len(items), 3)
            self.assertTrue(all(item["pseudo_label"] for item in items))
            self.assertTrue(all("label_sha256" not in item for item in items))
            self.assertTrue(all("label_bytes" not in item for item in items))

            payload = json.loads(manifest.read_text())
            payload["training"][0]["pseudo_label"] = False
            manifest.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "pseudo_label=true"):
                load_positive_pseudo_datalist(manifest, root, repeat=1)

    def test_pseudo_provenance_reaches_crop_metadata(self):
        dataset = EmptyCropTrustDataset(
            [{"image": "image", "label": "label", "pseudo_label": True}],
            trusted_empty=False,
            region="abdomen",
        )
        item = dataset[0]
        self.assertTrue(bool(item["positive_only_label"]))
        self.assertFalse(bool(item["complete_label"]))

    def test_mixed_labeled_and_pseudo_samples_collate(self):
        dataset = EmptyCropTrustDataset(
            [
                {
                    "image": torch.zeros(1, 4, 4, 4),
                    "label": torch.ones(1, 4, 4, 4),
                    "name": "pseudo",
                    "pseudo_label": True,
                },
                {
                    "image": torch.zeros(1, 4, 4, 4),
                    "label": torch.ones(1, 4, 4, 4),
                    "name": "labeled",
                },
            ],
            trusted_empty=False,
            region="abdomen",
        )
        batch = list_data_collate([dataset[0], dataset[1]])
        self.assertNotIn("pseudo_label", batch)
        self.assertTrue(
            torch.equal(
                batch["positive_only_label"],
                torch.tensor([True, False], dtype=torch.bool),
            )
        )


if __name__ == "__main__":
    unittest.main()
