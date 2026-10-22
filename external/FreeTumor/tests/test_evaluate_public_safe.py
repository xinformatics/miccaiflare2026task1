import tempfile
import unittest
from pathlib import Path

import torch

from external.FreeTumor.evaluate_public_safe import (
    case_output_name,
    shard_inventory,
    tumor_mask_from_logits,
)
import external.FreeTumor.evaluate_public_safe as evaluator


class EvaluatePublicSafeTest(unittest.TestCase):
    def test_case_output_name_strips_only_challenge_channel_suffix(self):
        self.assertEqual(
            case_output_name("FLARE23Ts_0042_0000.nii.gz"),
            "FLARE23Ts_0042.nii.gz",
        )
        with self.assertRaisesRegex(ValueError, "unexpected challenge image name"):
            case_output_name("FLARE23Ts_0042.nii.gz")

    def test_shard_inventory_is_disjoint_and_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = [f"FLARE23Ts_{index:04d}_0000.nii.gz" for index in range(7)]
            for name in reversed(names):
                (root / name).touch()

            shard0 = shard_inventory(root, 0, 2)
            shard1 = shard_inventory(root, 1, 2)

            self.assertFalse(set(shard0) & set(shard1))
            self.assertEqual(sorted(path.name for path in shard0 + shard1), names)
            with self.assertRaisesRegex(ValueError, "invalid shard"):
                shard_inventory(root, 2, 2)

    def test_tumor_mask_uses_only_released_tumor_channels(self):
        labels = torch.tensor([[[[0, 1, 2, 3, 4, 5, 6]]]])
        logits = torch.nn.functional.one_hot(labels, num_classes=7).movedim(-1, 1).float()
        mask = tumor_mask_from_logits(logits)
        self.assertEqual(mask.dtype, torch.uint8)
        self.assertEqual(mask.flatten().tolist(), [0, 0, 1, 0, 1, 0, 1])

    def test_tumor_mask_rejects_wrong_output_contract(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            tumor_mask_from_logits(torch.zeros((1, 2, 4, 4, 4)))

    def test_dataset_role_is_explicit_and_label_input_is_unavailable(self):
        source = Path(evaluator.__file__).read_text(encoding="utf-8")
        self.assertIn('choices=("public", "hidden")', source)
        self.assertIn('args.dataset_role == "hidden"', source)
        self.assertNotIn('add_argument("--labels"', source)


if __name__ == "__main__":
    unittest.main()
