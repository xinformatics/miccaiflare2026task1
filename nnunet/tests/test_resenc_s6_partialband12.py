import unittest
from pathlib import Path

import torch

from nnunet.resenc_s6_partialband12 import (
    BAND_INITIAL_LR,
    BAND_STAGE_END_EPOCHS,
    BAND_TRAINING_HORIZON,
    nnUNetTrainerResEncS6PanCancerPartialBand12,
    validated_parent_contract,
)
from nnunet.run_resenc_s6_partialband12 import resolve_trainer


class ResEncS6PartialBand12Test(unittest.TestCase):
    def test_runner_resolves_only_the_isolated_trainer(self):
        self.assertIs(
            resolve_trainer(None, "nnUNetTrainerResEncS6PanCancerPartialBand12", None),
            nnUNetTrainerResEncS6PanCancerPartialBand12,
        )

    def test_recipe_is_material_and_retains_recall_loss(self):
        trainer = object.__new__(nnUNetTrainerResEncS6PanCancerPartialBand12)
        trainer.enable_deep_supervision = False
        loss = trainer._build_loss()
        self.assertEqual(loss.alpha, 0.25)
        self.assertEqual(loss.beta, 0.75)
        self.assertEqual(BAND_INITIAL_LR, 1e-4)
        self.assertEqual(BAND_TRAINING_HORIZON, 100)
        self.assertEqual(BAND_STAGE_END_EPOCHS, (25, 50, 100))
        self.assertEqual(trainer.landmark_epochs, (25, 50))
        self.assertEqual(trainer.expected_parameter_count, 107_467_050)
        self.assertEqual(trainer.expected_global_batch_size, 16)

    def test_partial_target_uses_shape_band_not_component_box(self):
        trainer = object.__new__(nnUNetTrainerResEncS6PanCancerPartialBand12)
        trainer.full_margin_voxels = (1, 1, 1)
        target = torch.zeros((1, 1, 7, 10, 10), dtype=torch.int16)
        target[0, 0, 3, 2, 2:7] = 1
        target[0, 0, 3, 2:7, 6] = 1
        converted = trainer._partial_targets(target)
        self.assertEqual(int(converted[0, 0, 3, 6, 2]), 2)
        self.assertEqual(int(converted[0, 0, 3, 3, 3]), 0)

    def test_parent_contract_is_fail_closed(self):
        digest = "a" * 64
        self.assertEqual(
            validated_parent_contract(
                {
                    "RESENC_BAND12_PARENT_EPOCH": "300",
                    "RESENC_BAND12_PARENT_SHA256": digest,
                }
            ),
            (300, digest),
        )
        with self.assertRaises(ValueError):
            validated_parent_contract({})
        with self.assertRaises(ValueError):
            validated_parent_contract(
                {
                    "RESENC_BAND12_PARENT_EPOCH": "275",
                    "RESENC_BAND12_PARENT_SHA256": digest,
                }
            )
        with self.assertRaises(ValueError):
            validated_parent_contract(
                {
                    "RESENC_BAND12_PARENT_EPOCH": "300",
                    "RESENC_BAND12_PARENT_SHA256": "not-a-hash",
                }
            )

    def test_stage_environment_does_not_reuse_base_stage_variable(self):
        source = (
            Path(__file__).resolve().parent.parent / "resenc_s6_partialband12.py"
        ).read_text(encoding="utf-8")
        self.assertIn("RESENC_BAND12_STAGE_END_EPOCH", source)
        self.assertNotIn("Validation-Public", source)
        self.assertNotIn("Validation-Hidden", source)
        self.assertNotIn("codabench", source.lower())


if __name__ == "__main__":
    unittest.main()
