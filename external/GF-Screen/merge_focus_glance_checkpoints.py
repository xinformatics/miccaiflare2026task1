#!/usr/bin/env python3
"""Merge independently adapted GF-Screen Focus and Glance modules.

The Focus source supplies every ``segmentation_model.*`` tensor and the Glance
source supplies every ``glance_model.*`` tensor.  This is intended for the
paired experiments where the opposite module was frozen.  The output is an
inference checkpoint; optimizer and scheduler states are deliberately removed
so it cannot be mistaken for a resumable training state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


MODULE_PREFIXES = ("glance_model.", "segmentation_model.")
BOX_HEAD_PREFIX = "glance_model.box_layer."
CENTER_HEAD_PREFIX = "glance_model.center_layer."


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint is not a dictionary: {path}")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"checkpoint has no state_dict: {path}")
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise TypeError(f"state_dict contains non-tensor entries: {path}")
    return checkpoint, state


def module_keys(state: dict[str, torch.Tensor], prefix: str) -> set[str]:
    return {key for key in state if key.startswith(prefix)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--focus", type=Path, required=True)
    parser.add_argument("--glance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument(
        "--drop-training-box-head",
        action="store_true",
        help=(
            "Allow the Glance source to contain only the optional "
            "glance_model.box_layer.* training head missing from the Focus "
            "source. The head is omitted from the deployment checkpoint."
        ),
    )
    parser.add_argument(
        "--drop-training-center-head",
        action="store_true",
        help=(
            "Allow the Glance source to contain only the optional "
            "glance_model.center_layer.* training head missing from the Focus "
            "source. The head is omitted from the deployment checkpoint."
        ),
    )
    args = parser.parse_args()

    focus_checkpoint, focus_state = load_checkpoint(args.focus)
    glance_checkpoint, glance_state = load_checkpoint(args.glance)

    focus_state_keys = set(focus_state)
    glance_state_keys = set(glance_state)
    dropped_glance_keys: list[str] = []
    missing_focus_set = glance_state_keys - focus_state_keys
    allowed_training_prefixes = []
    if args.drop_training_box_head:
        allowed_training_prefixes.append(BOX_HEAD_PREFIX)
    if args.drop_training_center_head:
        allowed_training_prefixes.append(CENTER_HEAD_PREFIX)
    if missing_focus_set and allowed_training_prefixes and all(
        key.startswith(tuple(allowed_training_prefixes))
        for key in missing_focus_set
    ):
        dropped_glance_keys = sorted(missing_focus_set)
        glance_state_keys -= missing_focus_set

    if focus_state_keys != glance_state_keys:
        missing_focus = sorted(glance_state_keys - focus_state_keys)
        missing_glance = sorted(set(focus_state) - set(glance_state))
        raise ValueError(
            "source state_dict keys differ: "
            f"missing from Focus={missing_focus[:5]}, "
            f"missing from Glance={missing_glance[:5]}"
        )

    for key in focus_state:
        if focus_state[key].shape != glance_state[key].shape:
            raise ValueError(
                f"shape mismatch for {key}: "
                f"{tuple(focus_state[key].shape)} != {tuple(glance_state[key].shape)}"
            )

    glance_keys = module_keys(focus_state, "glance_model.")
    focus_keys = module_keys(focus_state, "segmentation_model.")
    unknown = set(focus_state) - glance_keys - focus_keys
    if not glance_keys or not focus_keys:
        raise ValueError("both Glance and Focus tensors must be present")
    if unknown:
        raise ValueError(f"unclassified model tensors: {sorted(unknown)[:10]}")

    merged_state = {
        key: (glance_state[key].detach().clone() if key in glance_keys else value.detach().clone())
        for key, value in focus_state.items()
    }
    output_checkpoint = {
        "state_dict": merged_state,
        "epoch": max(
            int(focus_checkpoint.get("epoch", 0)),
            int(glance_checkpoint.get("epoch", 0)),
        ),
        "best_acc": float(focus_checkpoint.get("best_acc", 0.0)),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    torch.save(output_checkpoint, temporary)
    temporary.replace(args.output)

    provenance = {
        "focus_checkpoint": str(args.focus.resolve()),
        "focus_sha256": file_sha256(args.focus),
        "glance_checkpoint": str(args.glance.resolve()),
        "glance_sha256": file_sha256(args.glance),
        "output_checkpoint": str(args.output.resolve()),
        "output_sha256": file_sha256(args.output),
        "focus_tensor_count": len(focus_keys),
        "glance_tensor_count": len(glance_keys),
        "dropped_training_glance_keys": dropped_glance_keys,
        "inference_only": True,
    }
    provenance_path = args.provenance or args.output.with_suffix(
        args.output.suffix + ".provenance.json"
    )
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
