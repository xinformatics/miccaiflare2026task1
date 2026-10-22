from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "merge_focus_glance_checkpoints.py"


def checkpoint(glance_value: float, focus_value: float, epoch: int) -> dict:
    return {
        "state_dict": {
            "glance_model.weight": torch.tensor([glance_value]),
            "glance_model.counter": torch.tensor(2, dtype=torch.int64),
            "segmentation_model.weight": torch.tensor([focus_value]),
            "segmentation_model.bias": torch.tensor([-focus_value]),
        },
        "epoch": epoch,
        "best_acc": 0.5,
        "optimizer_seg": {"must": "not survive"},
    }


class MergeFocusGlanceCheckpointTests(unittest.TestCase):
    def test_merges_exact_frozen_modules_and_writes_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            focus = tmp_path / "focus.pt"
            glance = tmp_path / "glance.pt"
            output = tmp_path / "merged.pt"
            torch.save(checkpoint(glance_value=1, focus_value=11, epoch=283), focus)
            torch.save(checkpoint(glance_value=7, focus_value=99, epoch=285), glance)

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--focus",
                    str(focus),
                    "--glance",
                    str(glance),
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            merged = torch.load(output, map_location="cpu", weights_only=False)
            self.assertEqual(set(merged), {"state_dict", "epoch", "best_acc"})
            self.assertEqual(merged["epoch"], 285)
            self.assertEqual(merged["state_dict"]["glance_model.weight"].item(), 7)
            self.assertEqual(merged["state_dict"]["segmentation_model.weight"].item(), 11)
            self.assertEqual(merged["state_dict"]["segmentation_model.bias"].item(), -11)

            provenance = json.loads(
                output.with_suffix(".pt.provenance.json").read_text(encoding="utf-8")
            )
            self.assertEqual(provenance["focus_tensor_count"], 2)
            self.assertEqual(provenance["glance_tensor_count"], 2)
            self.assertEqual(provenance["dropped_training_glance_keys"], [])
            self.assertIs(provenance["inference_only"], True)
            self.assertEqual(len(provenance["output_sha256"]), 64)

    def test_option_drops_only_training_box_head(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            focus = tmp_path / "focus.pt"
            glance = tmp_path / "glance.pt"
            output = tmp_path / "merged.pt"
            torch.save(checkpoint(glance_value=1, focus_value=11, epoch=283), focus)
            glance_checkpoint = checkpoint(glance_value=7, focus_value=99, epoch=285)
            glance_checkpoint["state_dict"]["glance_model.box_layer.weight"] = torch.tensor([13.0])
            glance_checkpoint["state_dict"]["glance_model.box_layer.bias"] = torch.tensor([17.0])
            torch.save(glance_checkpoint, glance)

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--focus",
                    str(focus),
                    "--glance",
                    str(glance),
                    "--output",
                    str(output),
                    "--drop-training-box-head",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            merged = torch.load(output, map_location="cpu", weights_only=False)
            self.assertNotIn("glance_model.box_layer.weight", merged["state_dict"])
            self.assertNotIn("glance_model.box_layer.bias", merged["state_dict"])
            self.assertEqual(merged["state_dict"]["glance_model.weight"].item(), 7)
            self.assertEqual(merged["state_dict"]["segmentation_model.weight"].item(), 11)
            provenance = json.loads(
                output.with_suffix(".pt.provenance.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                provenance["dropped_training_glance_keys"],
                [
                    "glance_model.box_layer.bias",
                    "glance_model.box_layer.weight",
                ],
            )

    def test_drop_option_rejects_unrelated_extra_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            focus = tmp_path / "focus.pt"
            glance = tmp_path / "glance.pt"
            output = tmp_path / "merged.pt"
            torch.save(checkpoint(glance_value=1, focus_value=11, epoch=283), focus)
            incompatible = checkpoint(glance_value=7, focus_value=99, epoch=285)
            incompatible["state_dict"]["glance_model.unexpected.weight"] = torch.tensor([5.0])
            torch.save(incompatible, glance)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--focus",
                    str(focus),
                    "--glance",
                    str(glance),
                    "--output",
                    str(output),
                    "--drop-training-box-head",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("source state_dict keys differ", result.stderr)
            self.assertFalse(output.exists())

    def test_options_drop_both_training_localization_heads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            focus = tmp_path / "focus.pt"
            glance = tmp_path / "glance.pt"
            output = tmp_path / "merged.pt"
            torch.save(checkpoint(glance_value=1, focus_value=11, epoch=283), focus)
            glance_checkpoint = checkpoint(glance_value=7, focus_value=99, epoch=285)
            for head, value in (("box_layer", 13.0), ("center_layer", 19.0)):
                glance_checkpoint["state_dict"][
                    f"glance_model.{head}.weight"
                ] = torch.tensor([value])
                glance_checkpoint["state_dict"][
                    f"glance_model.{head}.bias"
                ] = torch.tensor([-value])
            torch.save(glance_checkpoint, glance)

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--focus",
                    str(focus),
                    "--glance",
                    str(glance),
                    "--output",
                    str(output),
                    "--drop-training-box-head",
                    "--drop-training-center-head",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            merged = torch.load(output, map_location="cpu", weights_only=False)
            self.assertFalse(
                any(
                    "box_layer" in key or "center_layer" in key
                    for key in merged["state_dict"]
                )
            )
            provenance = json.loads(
                output.with_suffix(".pt.provenance.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                provenance["dropped_training_glance_keys"],
                [
                    "glance_model.box_layer.bias",
                    "glance_model.box_layer.weight",
                    "glance_model.center_layer.bias",
                    "glance_model.center_layer.weight",
                ],
            )

    def test_rejects_incompatible_state_dicts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            focus = tmp_path / "focus.pt"
            glance = tmp_path / "glance.pt"
            output = tmp_path / "merged.pt"
            torch.save(checkpoint(glance_value=1, focus_value=11, epoch=283), focus)
            incompatible = checkpoint(glance_value=7, focus_value=99, epoch=285)
            incompatible["state_dict"].pop("segmentation_model.bias")
            torch.save(incompatible, glance)

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--focus",
                    str(focus),
                    "--glance",
                    str(glance),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("source state_dict keys differ", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
