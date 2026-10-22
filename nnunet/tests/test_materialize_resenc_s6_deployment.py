import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from nnunet.materialize_resenc_s6_deployment import (  # noqa: E402
    DEPLOYMENTS,
    deployment_mask,
)


class MaterializeResEncS6DeploymentTests(unittest.TestCase):
    def test_standalone_is_exact_p50_mask(self):
        shape = (5, 5, 5)
        anchor = np.zeros(shape, bool)
        resenc = np.zeros(shape, bool)
        resenc[1:3, 1:3, 1:3] = True
        output = deployment_mask(
            "resenc_standalone_p50",
            anchor,
            resenc,
            anchor,
            anchor.astype(float),
            1.0,
        )
        self.assertTrue(np.array_equal(output, resenc))

    def test_confirmation_keeps_whole_anchor_component(self):
        shape = (8, 8, 8)
        anchor = np.zeros(shape, bool)
        anchor[1:3, 1:3, 1:3] = True
        anchor[5:7, 5:7, 5:7] = True
        probability = np.zeros(shape, np.float32)
        probability[1, 1, 1] = 0.8
        output = deployment_mask(
            "gf_anchor_confirm_either_p70",
            anchor,
            np.zeros(shape, bool),
            np.zeros(shape, bool),
            probability,
            1.0,
        )
        self.assertEqual(int(output.sum()), 8)
        self.assertTrue(output[2, 2, 2])
        self.assertFalse(output[5, 5, 5])

    def test_large_supported_addition_is_unioned_but_small_is_rejected(self):
        shape = (20, 20, 20)
        anchor = np.zeros(shape, bool)
        anchor[0, 0, 0] = True
        probability = np.zeros(shape, np.float32)
        probability[2:17, 2:17, 2:17] = 0.8  # 3375 mm3 at 1 mm3/voxel
        probability[18:20, 18:20, 18:20] = 0.8
        support = np.zeros(shape, bool)
        support[2:10, 2:10, 2:10] = True
        output = deployment_mask(
            "gf_union_resenc_p40_ftu_s10_mm3000",
            anchor,
            probability > 0.5,
            support,
            probability,
            1.0,
        )
        self.assertTrue(output[0, 0, 0])
        self.assertTrue(output[10, 10, 10])
        self.assertFalse(output[19, 19, 19])

    def test_boundary_refinement_replaces_shape_only_inside_three_mm_band(self):
        shape = (14, 14, 14)
        anchor = np.zeros(shape, bool)
        anchor[2:6, 2:6, 2:6] = True
        anchor[10:12, 10:12, 10:12] = True
        probability = np.zeros(shape, np.float32)
        probability[4:9, 4:6, 4:6] = 0.6
        probability[4, 4, 4] = 0.8
        freetumor = np.zeros(shape, bool)
        freetumor[10, 10, 10] = True
        output = deployment_mask(
            "gf_resenc_boundary3mm_p70p50_ftu",
            anchor,
            probability > 0.5,
            freetumor,
            probability,
            1.0,
            (1.0, 1.0, 1.0),
        )
        self.assertFalse(output[2, 2, 2])  # >3 mm from the learned p50 shape.
        self.assertTrue(output[3, 4, 4])   # Incumbent boundary near learned shape.
        self.assertTrue(output[8, 4, 4])   # Learned extension within 3 mm of anchor.
        self.assertEqual(int(output[10:12, 10:12, 10:12].sum()), 8)

    def test_boundary_refinement_uses_physical_not_voxel_radius(self):
        anchor = np.zeros((5, 5, 5), bool)
        anchor[1, 2, 2] = True
        probability = np.zeros(anchor.shape, np.float32)
        probability[1, 2, 2] = 0.8
        probability[2, 2, 2] = 0.6
        output = deployment_mask(
            "gf_resenc_boundary3mm_p70p50_ftu",
            anchor,
            probability > 0.5,
            np.zeros_like(anchor),
            probability,
            8.0,
            (4.0, 2.0, 1.0),
        )
        # The p50 extension is four physical millimetres away along axis 0, so
        # it is not added despite being one array voxel from a confirmed anchor.
        self.assertTrue(output[1, 2, 2])
        self.assertFalse(output[2, 2, 2])

    def test_unknown_deployment_fails_closed(self):
        array = np.zeros((2, 2, 2), bool)
        with self.assertRaises(ValueError):
            deployment_mask("unknown", array, array, array, array.astype(float), 1.0)


if __name__ == "__main__":
    unittest.main()
