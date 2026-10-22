import csv
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from nnunet.audit_amos_pseudo_anatomy import is_target_eligible
from nnunet.build_dataset503_organ_target import filtered_label, selected_components


class Dataset503OrganTargetTests(unittest.TestCase):
    def test_frozen_target_policy_accepts_only_plausible_focal_components(self):
        self.assertTrue(is_target_eligible(7, 1_000.0, 0.60, 0.20))
        self.assertTrue(is_target_eligible(14, 50_000.0, 0.95, 0.01))
        self.assertFalse(is_target_eligible(6, 10_000.0, 0.95, 0.01))
        self.assertFalse(is_target_eligible(7, 999.9, 0.95, 0.01))
        self.assertFalse(is_target_eligible(7, 50_000.1, 0.95, 0.01))
        self.assertFalse(is_target_eligible(7, 10_000.0, 0.599, 0.01))
        self.assertFalse(is_target_eligible(7, 10_000.0, 0.95, 0.201))

    def test_component_table_selection_is_deduplicated_and_sorted(self):
        with tempfile.TemporaryDirectory() as temporary:
            table = Path(temporary) / "components.csv"
            fields = ["case", "component", "dominant_organ", "target_eligible"]
            rows = [
                {"case": "amos0013", "component": 5, "dominant_organ": "stomach", "target_eligible": True},
                {"case": "amos0013", "component": 2, "dominant_organ": "stomach", "target_eligible": "1"},
                {"case": "amos0013", "component": 5, "dominant_organ": "stomach", "target_eligible": "true"},
                {"case": "amos0399", "component": 1, "dominant_organ": "urinary_bladder", "target_eligible": True},
                {"case": "amos0001", "component": 8, "dominant_organ": "liver", "target_eligible": False},
            ]
            with table.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            self.assertEqual(
                selected_components(table),
                {"amos0013": (2, 5), "amos0399": (1,)},
            )

    def test_filtered_label_keeps_exact_selected_six_connected_components(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "pseudo.nii.gz"
            destination = root / "target.nii.gz"
            mask = np.zeros((12, 12, 12), dtype=np.uint8)
            mask[1:3, 1:3, 1:3] = 1
            mask[6:9, 6:9, 6:9] = 1
            nib.save(nib.Nifti1Image(mask, np.diag([2.0, 2.0, 2.0, 1.0])), source)
            stats = filtered_label(source, (2,), destination)
            observed = np.asanyarray(nib.load(destination).dataobj)
            self.assertEqual(int(observed.sum()), 27)
            self.assertEqual(stats["voxels"], 27)
            self.assertAlmostEqual(stats["volume_mm3"], 216.0)
            self.assertFalse(observed[1:3, 1:3, 1:3].any())
            self.assertTrue(observed[6:9, 6:9, 6:9].all())


if __name__ == "__main__":
    unittest.main()
