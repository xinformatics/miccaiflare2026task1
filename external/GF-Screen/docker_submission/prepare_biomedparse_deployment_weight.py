#!/usr/bin/env python3
"""Strip BiomedParse optimizer state into a hash-bound inference checkpoint."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import torch


SOURCE_SHA256 = "6ea0bee43b983a7490e0b48689add96440c4405ca4b50b9f2fe21d5d03fd9440"
EXPECTED_TENSORS = 1052
EXPECTED_ELEMENTS = 371_798_786


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_payload(path: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    # mmap keeps the multi-GB training checkpoint from being copied wholesale
    # into anonymous RAM while the active segmentation trainer is running.
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state = payload.get("state_dict", payload)
    if not isinstance(payload, dict) or not isinstance(state, Mapping) or not state:
        raise RuntimeError(f"BiomedParse checkpoint has no state_dict: {path}")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
        raise RuntimeError("BiomedParse state_dict contains a non-tensor entry")
    return payload, dict(state)


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
        "--source",
        type=Path,
        default=project / "external/BiomedParse/container_workspace/model_weights/biomedparse_3D_AllData_MultiView_edge.ckpt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script.parent / "assets/biomedparse_3d_state_only.pt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if sha256(source) != SOURCE_SHA256:
        raise RuntimeError("BiomedParse source checkpoint SHA-256 mismatch")
    payload, state = load_payload(source)
    elements = sum(value.numel() for value in state.values())
    tensor_bytes = sum(value.numel() * value.element_size() for value in state.values())
    if len(state) != EXPECTED_TENSORS or elements != EXPECTED_ELEMENTS:
        raise RuntimeError("unexpected BiomedParse state tensor inventory")
    expected_training_keys = {
        "epoch", "global_step", "pytorch-lightning_version", "state_dict",
        "loops", "callbacks", "optimizer_states", "lr_schedulers",
        "hparams_name", "hyper_parameters",
    }
    if set(payload) != expected_training_keys or not payload.get("optimizer_states"):
        raise RuntimeError("unexpected BiomedParse Lightning checkpoint structure")

    atomic_torch_save(
        {
            "schema_version": 1,
            "role": "biomedparse_3d_inference_state_only",
            "source_sha256": SOURCE_SHA256,
            "state_dict": state,
        },
        output,
    )
    compact_payload, compact_state = load_payload(output)
    if (
        compact_payload.get("role") != "biomedparse_3d_inference_state_only"
        or compact_payload.get("source_sha256") != SOURCE_SHA256
        or compact_state.keys() != state.keys()
    ):
        raise RuntimeError("serialized BiomedParse deployment provenance mismatch")
    for key in state:
        if not torch.equal(state[key], compact_state[key]):
            raise RuntimeError(f"serialized BiomedParse tensor differs: {key}")

    source_bytes = source.stat().st_size
    output_bytes = output.stat().st_size
    manifest = {
        "schema_version": 1,
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "transformation": "remove Lightning optimizer, loops, callbacks, scheduler and hyperparameter training state",
        "weights_changed": False,
        "source": {"path": str(source), "bytes": source_bytes, "sha256": SOURCE_SHA256},
        "deployment": {"path": str(output), "bytes": output_bytes, "sha256": sha256(output)},
        "tensor_inventory": {
            "tensors": len(state),
            "elements": elements,
            "unserialized_tensor_bytes": tensor_bytes,
        },
        "bytes": {
            "saved": source_bytes - output_bytes,
            "fraction_saved": (source_bytes - output_bytes) / source_bytes,
        },
        "public_images_read": False,
        "public_labels_read": False,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
        "deployment_authorized": False,
    }
    manifest_path = output.with_suffix(".json")
    atomic_json(manifest, manifest_path)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
