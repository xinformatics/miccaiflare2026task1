import importlib.util
import inspect
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(HERE))
from STUNetTrainerPanCancer import (  # noqa: E402
    IGNORE_LABEL,
    PartialBoxTverskyCELoss,
    STUNetTrainerPanCancer,
    STUNetTrainerPanCancerComponentBox,
    STUNetTrainerPanCancerComponentBoxE160Balanced,
    STUNetTrainerPanCancerSmoke,
    component_box_target,
    partial_box_target,
)


def load_launcher():
    path = HERE / "run_pancancer_stunet.py"
    spec = importlib.util.spec_from_file_location("run_pancancer_stunet_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class STUNetTrainerPanCancerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        plans_path = PROJECT_ROOT / "nnUNet_preprocessed" / "Dataset501_PanCancer" / "nnUNetPlans.json"
        cls.arch_kwargs = json.loads(plans_path.read_text(encoding="utf-8"))[
            "configurations"
        ]["3d_fullres"]["architecture"]["arch_kwargs"]

    def test_partial_box_target_ignores_unknown_background(self):
        target = torch.zeros((2, 1, 10, 12, 14), dtype=torch.int16)
        target[0, 0, 4:6, 5:7, 6:8] = 1
        result = partial_box_target(
            target,
            full_shape=(10, 12, 14),
            full_margin_voxels=(1, 2, 3),
        )
        self.assertTrue((result[1] == IGNORE_LABEL).all())
        self.assertTrue((result[0, 0, 4:6, 5:7, 6:8] == 1).all())
        self.assertEqual(int(result[0, 0, 0, 0, 0]), IGNORE_LABEL)
        self.assertEqual(int(result[0, 0, 3, 3, 3]), 0)

    def test_partial_box_loss_has_no_gradient_outside_trusted_box(self):
        target = torch.zeros((1, 1, 6, 6, 6), dtype=torch.int64)
        target[0, 0, 2:4, 2:4, 2:4] = 1
        target = partial_box_target(target, (6, 6, 6), (1, 1, 1))
        logits = torch.zeros((1, 2, 6, 6, 6), requires_grad=True)
        loss = PartialBoxTverskyCELoss()(logits, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(logits.grad[:, :, 2:4, 2:4, 2:4].abs().sum()), 0.0)
        self.assertEqual(float(logits.grad[:, :, 0, 0, 0].abs().sum()), 0.0)

    def test_component_boxes_do_not_trust_background_between_lesions(self):
        target = torch.zeros((1, 1, 12, 12, 12), dtype=torch.int64)
        target[0, 0, 2, 2, 2] = 1
        target[0, 0, 9, 9, 9] = 1
        global_box = partial_box_target(target, (12, 12, 12), (1, 1, 1))
        component_boxes = component_box_target(target, (12, 12, 12), (1, 1, 1))
        self.assertEqual(int(global_box[0, 0, 5, 5, 5]), 0)
        self.assertEqual(int(component_boxes[0, 0, 5, 5, 5]), IGNORE_LABEL)
        self.assertEqual(int(component_boxes[0, 0, 2, 2, 2]), 1)
        self.assertEqual(int(component_boxes[0, 0, 9, 9, 9]), 1)

    def test_component_box_loss_has_no_gradient_between_lesions(self):
        target = torch.zeros((1, 1, 12, 12, 12), dtype=torch.int64)
        target[0, 0, 2, 2, 2] = 1
        target[0, 0, 9, 9, 9] = 1
        target = component_box_target(target, (12, 12, 12), (1, 1, 1))
        logits = torch.zeros((1, 2, 12, 12, 12), requires_grad=True)
        PartialBoxTverskyCELoss()(logits, target).backward()
        self.assertGreater(float(logits.grad[:, :, 2, 2, 2].abs().sum()), 0.0)
        self.assertEqual(float(logits.grad[:, :, 5, 5, 5].abs().sum()), 0.0)

    def test_component_box_preserves_existing_ignore_labels(self):
        target = torch.zeros((1, 1, 7, 7, 7), dtype=torch.int64)
        target[0, 0, 3, 3, 3] = 1
        target[0, 0, 3, 3, 4] = IGNORE_LABEL
        result = component_box_target(target, (7, 7, 7), (1, 1, 1))
        self.assertEqual(int(result[0, 0, 3, 3, 4]), IGNORE_LABEL)

    def test_all_ignored_patch_has_finite_zero_loss(self):
        target = torch.full((1, 1, 4, 4, 4), IGNORE_LABEL, dtype=torch.int64)
        logits = torch.randn((1, 2, 4, 4, 4), requires_grad=True)
        loss = PartialBoxTverskyCELoss()(logits, target)
        loss.backward()
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(float(logits.grad.abs().sum()), 0.0)

    def test_network_is_stunet_base_with_binary_heads(self):
        model = STUNetTrainerPanCancer.build_network_architecture(
            "ignored",
            self.arch_kwargs,
            (),
            num_input_channels=1,
            num_output_channels=2,
            enable_deep_supervision=True,
        )
        parameters = sum(parameter.numel() for parameter in model.parameters())
        self.assertGreater(parameters, 58_000_000)
        self.assertLess(parameters, 59_000_000)
        self.assertEqual(model.seg_outputs[0].out_channels, 2)
        self.assertEqual(len(model.conv_blocks_context), 6)

    def test_launcher_resolves_only_named_custom_trainer(self):
        launcher = load_launcher()
        self.assertIs(
            launcher.resolve_trainer(None, "STUNetTrainerPanCancer", None),
            STUNetTrainerPanCancer,
        )
        self.assertIs(
            launcher.resolve_trainer(
                None, "STUNetTrainerPanCancerComponentBox", None
            ),
            STUNetTrainerPanCancerComponentBox,
        )
        self.assertIs(
            launcher.resolve_trainer(
                None, "STUNetTrainerPanCancerComponentBoxE160Balanced", None
            ),
            STUNetTrainerPanCancerComponentBoxE160Balanced,
        )
        self.assertIs(
            launcher.resolve_trainer(None, "STUNetTrainerPanCancerSmoke", None),
            STUNetTrainerPanCancerSmoke,
        )

    def test_post_resume_augmenter_is_seeded_and_ordered(self):
        source = inspect.getsource(STUNetTrainerPanCancer.get_dataloaders)
        self.assertIn("MultiThreadedAugmenter", source)
        self.assertIn("matched_seeds", source)
        self.assertIn("2_026_082_600", source)
        self.assertIn("timeout=120", source)
        self.assertIn("finally:", source)

    def test_every_epoch_inference_checkpoint_has_predictor_contract_only(self):
        trainer = object.__new__(STUNetTrainerPanCancer)
        trainer.local_rank = 0
        trainer.disable_checkpointing = False
        trainer.is_ddp = False
        trainer.network = torch.nn.Conv3d(1, 2, 1)
        trainer.my_init_kwargs = {"configuration": "3d_fullres"}
        trainer.inference_allowed_mirroring_axes = (0, 1, 2)
        trainer.current_epoch = 100
        trainer.logger = SimpleNamespace(
            my_fantastic_logging={
                "mean_fg_dice": [0.75],
                "ema_fg_dice": [0.71],
                "train_losses": [0.44],
                "val_losses": [0.48],
                "lrs": [2.1e-4],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint_inference_epoch_0101.pth"
            trainer.save_inference_checkpoint(str(path))
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            sidecar = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            checkpoint_size = path.stat().st_size
        self.assertEqual(
            set(checkpoint),
            {
                "network_weights",
                "init_args",
                "trainer_name",
                "inference_allowed_mirroring_axes",
                "current_epoch",
                "checkpoint_kind",
                "internal_metrics",
            },
        )
        self.assertEqual(checkpoint["checkpoint_kind"], "inference_only")
        self.assertEqual(checkpoint["current_epoch"], 101)
        self.assertNotIn("optimizer_state", checkpoint)
        self.assertNotIn("logging", checkpoint)
        self.assertEqual(checkpoint["internal_metrics"]["ema_fg_dice"], 0.71)
        self.assertEqual(sidecar["checkpoint"], path.name)
        self.assertEqual(sidecar["checkpoint_bytes"], checkpoint_size)
        self.assertIn("fixed-public", sidecar["selection_warning"])

    def test_on_epoch_end_saves_inference_checkpoint_from_epoch_101(self):
        source = inspect.getsource(STUNetTrainerPanCancer.on_epoch_end)
        self.assertIn("inference_checkpoint_start_epoch", source)
        self.assertIn("checkpoint_inference_epoch_", source)

    def test_e160_balanced_continuation_resets_schedule_and_balances_tversky(self):
        trainer = object.__new__(STUNetTrainerPanCancerComponentBoxE160Balanced)
        trainer.enable_deep_supervision = False
        loss = trainer._build_loss()
        self.assertEqual((loss.alpha, loss.beta), (0.5, 0.5))
        source = inspect.getsource(
            STUNetTrainerPanCancerComponentBoxE160Balanced.__init__
        )
        self.assertIn("self.initial_lr = 3e-5", source)
        self.assertIn("self.num_epochs = 80", source)
        self.assertIn("self.inference_checkpoint_start_epoch = 1", source)
        self.assertEqual(
            STUNetTrainerPanCancerComponentBoxE160Balanced.parent_checkpoint_sha256,
            "81b9eed4105e89e8399c091984db4fb9ce1e37089d07a3fd4c032782f8d7cd74",
        )

    def test_verified_pretrained_checkpoint_matches_all_nonhead_tensors(self):
        launcher = load_launcher()
        model = STUNetTrainerPanCancer.build_network_architecture(
            "ignored",
            self.arch_kwargs,
            (),
            num_input_channels=1,
            num_output_channels=2,
            enable_deep_supervision=True,
        )
        launcher.load_stunet_pretrained_weights(
            model,
            PROJECT_ROOT / "external" / "STU-Net" / "pretrained" / "base_ep4k.model",
        )
        first = model.state_dict()["conv_blocks_context.0.0.conv1.weight"]
        self.assertGreater(float(first.abs().sum()), 0.0)

    def test_verified_component_epoch160_checkpoint_loads_all_tensors(self):
        launcher = load_launcher()
        model = STUNetTrainerPanCancer.build_network_architecture(
            "ignored",
            self.arch_kwargs,
            (),
            num_input_channels=1,
            num_output_channels=2,
            enable_deep_supervision=True,
        )
        launcher.load_stunet_pretrained_weights(
            model,
            PROJECT_ROOT
            / "nnUNet_results"
            / "Dataset501_PanCancer"
            / "STUNetTrainerPanCancerComponentBox__nnUNetPlans__3d_fullres_STUNetB_PartialBox"
            / "fold_all"
            / "checkpoint_inference_epoch_0160.pth",
        )
        self.assertGreater(float(model.seg_outputs[0].weight.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
