import sys
import csv
import contextlib
import io
import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

import nibabel as nib
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biomedparse_fusion import (
    apply_config,
    apply_configs,
    confirmation_support_strength_configurations,
    configurations,
    run_screen,
    support_within_distance,
)


class BiomedParseFusionTests(unittest.TestCase):
    def setUp(self):
        self.base = np.zeros((9, 9, 9), dtype=bool)
        self.base[1:3, 1:3, 1:3] = True
        self.base[6:8, 6:8, 6:8] = True
        self.support = np.zeros_like(self.base)
        self.support[1:3, 1:3, 1:3] = True
        self.support[4:6, 4:6, 4:6] = True
        self.spacing = (1.0, 1.0, 1.0)

    def test_configuration_grid_is_fixed(self):
        configs = configurations()
        self.assertEqual(len(configs), 16)
        self.assertEqual({config["kind"] for config in configs}, {"standalone", "union", "confirm"})

    def test_standalone_filter(self):
        result = apply_config(
            self.base,
            self.support,
            self.spacing,
            {"kind": "standalone", "support_min_mm3": 8},
        )
        np.testing.assert_array_equal(result, self.support)

    def test_disjoint_union_adds_only_new_component(self):
        result = apply_config(
            self.base,
            self.support,
            self.spacing,
            {"kind": "union", "mode": "disjoint", "support_min_mm3": 1},
        )
        self.assertTrue(np.all(result[self.base]))
        self.assertTrue(bool(result[4, 4, 4]))

    def test_confirmation_removes_unsupported_component(self):
        result = apply_config(
            self.base,
            self.support,
            self.spacing,
            {"kind": "confirm", "distance_mm": 0, "rescue_min_mm3": "inf"},
        )
        self.assertTrue(bool(result[1, 1, 1]))
        self.assertFalse(bool(result[6, 6, 6]))

    def test_confirmation_can_require_physical_overlap(self):
        support = np.zeros_like(self.base)
        support[1, 1, 1] = True
        weak = apply_config(
            self.base,
            support,
            self.spacing,
            {
                "kind": "confirm",
                "distance_mm": 0,
                "rescue_min_mm3": "inf",
                "min_support_overlap_mm3": 2,
            },
        )
        exact = apply_config(
            self.base,
            support,
            self.spacing,
            {"kind": "confirm", "distance_mm": 0, "rescue_min_mm3": "inf"},
        )
        self.assertFalse(bool(weak[1, 1, 1]))
        self.assertTrue(bool(exact[1, 1, 1]))

    def test_confirmation_can_require_component_fraction(self):
        support = np.zeros_like(self.base)
        support[1, 1, 1] = True
        result = apply_config(
            self.base,
            support,
            self.spacing,
            {
                "kind": "confirm",
                "distance_mm": 0,
                "rescue_min_mm3": "inf",
                "min_support_overlap_fraction": 0.20,
            },
        )
        self.assertFalse(bool(result[1, 1, 1]))

    def test_strength_rescue_requires_some_support_and_large_component(self):
        support = np.zeros_like(self.base)
        support[1, 1, 1] = True
        result = apply_config(
            self.base,
            support,
            self.spacing,
            {
                "kind": "confirm",
                "distance_mm": 0,
                "rescue_min_mm3": "inf",
                "min_support_overlap_fraction": 0.20,
                "weak_support_rescue_min_mm3": 8,
            },
        )
        self.assertTrue(bool(result[1, 1, 1]))
        self.assertFalse(bool(result[6, 6, 6]))

    def test_support_strength_grid_is_bounded_and_cached_path_matches(self):
        configs = confirmation_support_strength_configurations()
        self.assertEqual(len(configs), 15)
        cached = apply_configs(self.base, self.support, self.spacing, configs)
        individual = [
            apply_config(self.base, self.support, self.spacing, config)
            for config in configs
        ]
        for left, right in zip(cached, individual):
            np.testing.assert_array_equal(left, right)

    def test_confirmation_rescues_large_component(self):
        result = apply_config(
            self.base,
            self.support,
            self.spacing,
            {"kind": "confirm", "distance_mm": 0, "rescue_min_mm3": 8},
        )
        np.testing.assert_array_equal(result, self.base)

    def test_confirmation_top_k_rescues_only_largest_eligible_component(self):
        base = np.zeros((12, 12, 12), dtype=bool)
        base[0, 0, 0] = True
        base[3:5, 3:5, 3:5] = True
        base[8:11, 8:11, 8:11] = True
        support = np.zeros_like(base)
        support[0, 0, 0] = True
        config = {
            "kind": "confirm",
            "distance_mm": 0,
            "rescue_min_mm3": "inf",
            "rescue_top_k": 1,
            "rescue_top_k_min_mm3": 1,
            "rescue_top_k_max_mm3": 100,
        }
        result = apply_config(base, support, self.spacing, config)
        self.assertTrue(bool(result[0, 0, 0]))
        self.assertFalse(bool(result[3, 3, 3]))
        self.assertTrue(bool(result[8, 8, 8]))
        cached = apply_configs(base, support, self.spacing, [config])[0]
        np.testing.assert_array_equal(result, cached)

    def test_cached_grid_matches_individual_application(self):
        configs = configurations()
        cached = apply_configs(self.base, self.support, self.spacing, configs)
        individual = [
            apply_config(self.base, self.support, self.spacing, config)
            for config in configs
        ]
        for left, right in zip(cached, individual):
            np.testing.assert_array_equal(left, right)

    def test_anisotropic_dilation_matches_euclidean_distance_transform(self):
        from scipy import ndimage

        support = np.zeros((13, 15, 9), dtype=bool)
        support[2, 8, 4] = True
        support[10, 3, 6] = True
        spacing = (0.8, 1.7, 3.5)
        expected = (
            ndimage.distance_transform_edt(~support, sampling=spacing) <= 5.0 + 1e-6
        )
        actual = support_within_distance(support, spacing, 5.0)
        np.testing.assert_array_equal(actual, expected)

    def test_end_to_end_screen_materializes_fixed_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_dir = root / "base"
            labels_dir = root / "labels"
            output_dir = root / "output"
            base_dir.mkdir()
            labels_dir.mkdir()
            supports = []
            for support_index in range(6):
                support_dir = root / f"support_{support_index}"
                support_dir.mkdir()
                supports.append(f"s{support_index}={support_dir}")
            for case_index in range(2):
                name = f"case_{case_index}.nii.gz"
                target = self.base.copy()
                target[4:6, 4:6, 4:6] = True
                nib.save(nib.Nifti1Image(self.base.astype(np.uint8), np.eye(4)), base_dir / name)
                nib.save(nib.Nifti1Image(target.astype(np.uint8), np.eye(4)), labels_dir / name)
                for support_spec in supports:
                    support_dir = Path(support_spec.split("=", 1)[1])
                    nib.save(
                        nib.Nifti1Image(self.support.astype(np.uint8), np.eye(4)),
                        support_dir / name,
                    )
            arguments = SimpleNamespace(
                base=base_dir,
                support=supports,
                labels=labels_dir,
                output=output_dir,
                workers=2,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                run_screen(arguments)
            report = json.loads((output_dir / "screen.json").read_text())
            self.assertEqual(len(report["results"]), 96)
            self.assertNotIn(b"\r", (output_dir / "shortlist.tsv").read_bytes())
            with (output_dir / "shortlist.tsv").open(newline="") as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            self.assertGreaterEqual(len(rows), 6)
            self.assertLessEqual(len(rows), 8)
            for row in rows:
                self.assertTrue(all(value for value in row.values()))
                predictions = Path(row["predictions"])
                self.assertEqual(len(list(predictions.glob("*.nii.gz"))), 2)

            confirm_output = root / "confirm_only"
            arguments.output = confirm_output
            arguments.kind = ["confirm"]
            with contextlib.redirect_stdout(io.StringIO()):
                run_screen(arguments)
            confirm_report = json.loads((confirm_output / "screen.json").read_text())
            self.assertEqual(confirm_report["declared_configurations_per_support"], 6)
            self.assertEqual(len(confirm_report["results"]), 36)
            self.assertTrue(
                all(row["kind"] == "confirm" for row in confirm_report["results"])
            )

            top_k_output = root / "confirm_top_k"
            arguments.output = top_k_output
            arguments.include_top_k_rescue = True
            with contextlib.redirect_stdout(io.StringIO()):
                run_screen(arguments)
            top_k_report = json.loads((top_k_output / "screen.json").read_text())
            self.assertEqual(top_k_report["declared_configurations_per_support"], 9)
            self.assertTrue(
                any("rescue_top_k" in tag for tag in top_k_report["shortlist_tags"])
            )


if __name__ == "__main__":
    unittest.main()
