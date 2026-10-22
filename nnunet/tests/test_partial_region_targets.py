import unittest
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANCANCER_ROOT = PROJECT_ROOT / "external/STU-Net/pancancer"
if str(PANCANCER_ROOT) not in sys.path:
    sys.path.insert(0, str(PANCANCER_ROOT))

from nnunet.audit_public_partial_region_policy import shape_band_metrics_from_slice
from nnunet.partial_region_targets import (
    IGNORE_LABEL,
    scaled_physical_margins,
    shape_linf_band_target,
)
from STUNetTrainerPanCancer import PartialBoxTverskyCELoss


class PartialRegionTargetsTest(unittest.TestCase):
    def test_deep_supervision_margin_scales_up_not_down_to_zero(self):
        self.assertEqual(
            scaled_physical_margins((5, 8, 8), (64, 80, 80), (128, 160, 160)),
            (3, 4, 4),
        )
        self.assertEqual(
            scaled_physical_margins((1, 1, 1), (1, 1, 1), (128, 160, 160)),
            (1, 1, 1),
        )

    def test_shape_band_preserves_binary_values_and_existing_ignore(self):
        target = torch.zeros((1, 1, 9, 9, 9), dtype=torch.int16)
        target[0, 0, 4, 4, 4] = 1
        target[0, 0, 4, 4, 5] = IGNORE_LABEL
        result = shape_linf_band_target(target, (9, 9, 9), (1, 1, 1))
        self.assertEqual(int(result[0, 0, 4, 4, 4]), 1)
        self.assertEqual(int(result[0, 0, 4, 4, 5]), IGNORE_LABEL)
        self.assertEqual(int(result[0, 0, 4, 3, 4]), 0)
        self.assertEqual(int(result[0, 0, 0, 0, 0]), IGNORE_LABEL)

    def test_empty_patch_is_fully_ignored(self):
        target = torch.zeros((2, 1, 5, 5, 5), dtype=torch.int16)
        result = shape_linf_band_target(target, (5, 5, 5), (1, 1, 1))
        self.assertTrue(torch.all(result == IGNORE_LABEL))

    def test_mask_matches_audited_scipy_geometry(self):
        labels = np.zeros((9, 12, 12), dtype=np.uint8)
        labels[4, 3, 3:9] = 1
        labels[4, 3:9, 8] = 1
        labels[4, 8, 3] = 2
        target = torch.from_numpy((labels == 1).astype(np.int16))[None, None]
        converted = shape_linf_band_target(target, labels.shape, (2, 2, 2))
        collision_from_target = bool(
            np.any((converted[0, 0].numpy() != IGNORE_LABEL) & (labels == 2))
        )
        audited = shape_band_metrics_from_slice(
            labels,
            1,
            (slice(4, 5), slice(3, 9), slice(3, 9)),
            int((labels == 1).sum()),
            int((labels > 0).sum()),
            (1.0, 1.0, 1.0),
            2.0,
        )
        self.assertEqual(collision_from_target, audited["any_collision"])

    def test_ignored_voxels_receive_exactly_zero_gradient(self):
        target = torch.zeros((1, 1, 9, 9, 9), dtype=torch.int16)
        target[0, 0, 4, 4, 4] = 1
        converted = shape_linf_band_target(target, (9, 9, 9), (1, 1, 1))
        logits = torch.zeros((1, 2, 9, 9, 9), requires_grad=True)
        loss = PartialBoxTverskyCELoss(ignore_label=IGNORE_LABEL)(logits, converted)
        loss.backward()
        self.assertGreater(float(logits.grad[:, :, 4, 4, 4].abs().sum()), 0.0)
        self.assertGreater(float(logits.grad[:, :, 4, 3, 4].abs().sum()), 0.0)
        self.assertEqual(float(logits.grad[:, :, 0, 0, 0].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
