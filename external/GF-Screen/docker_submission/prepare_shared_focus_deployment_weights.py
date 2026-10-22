#!/usr/bin/env python3
"""Create compact, hash-bound GF assets for the shared-Focus inference path.

The bbox checkpoint supplies the complete GF model (bbox Glance + shared Focus).
The CBG checkpoint contributes only its different Glance model. Optimizer and
scheduler state are deliberately omitted. Source checkpoints are never edited.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import torch


BBOX_SHA256 = "04a3cf7cb5209e65c253e6b1dd70ad7a972008e396825d5ede5ec3a933eb3fdb"
CBG_SHA256 = "3008615b57960e36b10b534b750acd4ecfec0f94950368bb80851d3734d4f190"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError(f"checkpoint has no state_dict: {path}")
    result: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise RuntimeError(f"non-tensor state entry in {path}: {key!r}")
        result[key] = value.detach().cpu()
    return result


def tensor_digest(state: Mapping[str, torch.Tensor], prefix: str) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    keys = sorted(key for key in state if key.startswith(prefix))
    elements = 0
    for key in keys:
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
        elements += tensor.numel()
    return digest.hexdigest(), len(keys), elements


def atomic_torch_save(payload: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(payload: dict, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    script = Path(__file__).resolve()
    project = script.parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bbox-source",
        type=Path,
        default=project / "external/GF-Screen/runs/gf_a055_bbox_pu_lr3e5_15ep_20260819/model.pt",
    )
    parser.add_argument(
        "--cbg-source",
        type=Path,
        default=project / "external/GF-Screen/runs/gf_glance_cbg_p4n1_ep280_5ep_20260822/checkpoint_epoch_0283.pt",
    )
    parser.add_argument("--output", type=Path, default=script.parent / "assets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bbox_source = args.bbox_source.resolve()
    cbg_source = args.cbg_source.resolve()
    if sha256(bbox_source) != BBOX_SHA256:
        raise RuntimeError("bbox source checkpoint SHA-256 mismatch")
    if sha256(cbg_source) != CBG_SHA256:
        raise RuntimeError("CBG source checkpoint SHA-256 mismatch")

    bbox_state = load_state(bbox_source)
    bbox_focus_digest, bbox_focus_keys, bbox_focus_elements = tensor_digest(
        bbox_state, "segmentation_model."
    )
    bbox_glance_keys = sum(key.startswith("glance_model.") for key in bbox_state)
    if bbox_focus_keys != 167 or bbox_glance_keys != 104 or len(bbox_state) != 271:
        raise RuntimeError("unexpected bbox GF tensor inventory")

    cbg_state = load_state(cbg_source)
    cbg_focus_digest, cbg_focus_keys, cbg_focus_elements = tensor_digest(
        cbg_state, "segmentation_model."
    )
    if (
        cbg_focus_digest != bbox_focus_digest
        or cbg_focus_keys != bbox_focus_keys
        or cbg_focus_elements != bbox_focus_elements
    ):
        raise RuntimeError("bbox and CBG Focus tensors are not bit-identical")
    cbg_glance = {
        key.removeprefix("glance_model."): value
        for key, value in cbg_state.items()
        if key.startswith("glance_model.")
    }
    if len(cbg_glance) != 104 or len(cbg_state) != 271:
        raise RuntimeError("unexpected CBG GF tensor inventory")

    bbox_output = args.output / "gf_bbox_full_state.pt"
    cbg_output = args.output / "gf_cbg_glance_state.pt"
    manifest_output = args.output / "shared_focus_assets.json"
    atomic_torch_save(
        {
            "schema_version": 1,
            "role": "bbox_glance_and_shared_focus",
            "source_sha256": BBOX_SHA256,
            "state_dict": bbox_state,
        },
        bbox_output,
    )
    atomic_torch_save(
        {
            "schema_version": 1,
            "role": "cbg_glance_only",
            "source_sha256": CBG_SHA256,
            "state_dict": cbg_glance,
        },
        cbg_output,
    )

    # Reload the serialized assets and require tensor-exact identity before
    # publishing their hashes in the manifest.
    bbox_reloaded = load_state(bbox_output)
    cbg_reloaded = load_state(cbg_output)
    if bbox_reloaded.keys() != bbox_state.keys() or not all(
        torch.equal(bbox_reloaded[key], bbox_state[key]) for key in bbox_state
    ):
        raise RuntimeError("serialized bbox deployment state differs from source")
    if cbg_reloaded.keys() != cbg_glance.keys() or not all(
        torch.equal(cbg_reloaded[key], cbg_glance[key]) for key in cbg_glance
    ):
        raise RuntimeError("serialized CBG Glance state differs from source")

    source_bytes = bbox_source.stat().st_size + cbg_source.stat().st_size
    output_bytes = bbox_output.stat().st_size + cbg_output.stat().st_size
    manifest = {
        "schema_version": 1,
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "transformation": "training-state removal plus exact shared-Focus deduplication",
        "weights_changed": False,
        "source_assets": {
            "bbox": {"path": str(bbox_source), "bytes": bbox_source.stat().st_size, "sha256": BBOX_SHA256},
            "cbg": {"path": str(cbg_source), "bytes": cbg_source.stat().st_size, "sha256": CBG_SHA256},
        },
        "deployment_assets": {
            "bbox_full": {"path": str(bbox_output.resolve()), "bytes": bbox_output.stat().st_size, "sha256": sha256(bbox_output)},
            "cbg_glance": {"path": str(cbg_output.resolve()), "bytes": cbg_output.stat().st_size, "sha256": sha256(cbg_output)},
        },
        "tensor_inventory": {
            "bbox_full": len(bbox_state),
            "bbox_glance": bbox_glance_keys,
            "shared_focus": bbox_focus_keys,
            "shared_focus_elements": bbox_focus_elements,
            "cbg_glance": len(cbg_glance),
            "shared_focus_digest": bbox_focus_digest,
        },
        "bytes": {
            "source_total": source_bytes,
            "deployment_total": output_bytes,
            "saved": source_bytes - output_bytes,
            "fraction_saved": (source_bytes - output_bytes) / source_bytes,
        },
        "public_images_read": False,
        "public_labels_read": False,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    atomic_json(manifest, manifest_output)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
