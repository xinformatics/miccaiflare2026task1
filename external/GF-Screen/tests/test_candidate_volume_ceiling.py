import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from analyze_candidate_volume_ceiling import analyze_case


class CandidateVolumeCeilingTests(unittest.TestCase):
    def test_maximum_volume_controls_recoverable_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target_path = root / "target.nii.gz"
            proposal_path = root / "proposal.nii.gz"
            affine = np.eye(4)
            target = np.zeros((7, 7, 7), dtype=np.uint8)
            target[1, 1, 1] = 1
            target[5, 5, 5] = 1
            proposal = np.zeros_like(target)
            proposal[1, 1, 1] = 1
            proposal[3:6, 5, 5] = 1
            nib.save(nib.Nifti1Image(target, affine), target_path)
            nib.save(nib.Nifti1Image(proposal, affine), proposal_path)

            row = analyze_case(
                (
                    "case.nii.gz",
                    str(target_path),
                    str(proposal_path),
                    0.0,
                    (1.0, 3.0),
                )
            )

            small = "max_000000000001"
            all_components = "max_000000000003"
            self.assertEqual(row[f"{small}_candidates"], 1)
            self.assertAlmostEqual(row[f"{small}_voxel_oracle_dsc"], 2 / 3)
            self.assertAlmostEqual(row[f"{small}_target_reconstruction_dsc"], 2 / 3)
            self.assertEqual(row[f"{all_components}_candidates"], 2)
            self.assertEqual(row[f"{all_components}_voxel_oracle_dsc"], 1.0)
            self.assertEqual(row[f"{all_components}_target_reconstruction_dsc"], 1.0)
            self.assertAlmostEqual(row[f"{all_components}_whole_component_dsc"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
