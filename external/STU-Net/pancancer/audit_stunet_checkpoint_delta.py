#!/usr/bin/env python3
"""Hash-bound, label-blind STU-Net checkpoint movement audit."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    weights = checkpoint.get("network_weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"checkpoint has no network_weights: {path}")
    return checkpoint


def blank() -> dict[str, float | int]:
    return {
        "tensors": 0,
        "elements": 0,
        "elements_changed": 0,
        "reference_sq": 0.0,
        "difference_sq": 0.0,
        "max_abs_delta": 0.0,
    }


def update(stats: dict[str, float | int], reference: torch.Tensor, candidate: torch.Tensor) -> None:
    stats["tensors"] += 1
    stats["elements"] += reference.numel()
    stats["elements_changed"] += int(torch.count_nonzero(reference != candidate))
    if not (reference.is_floating_point() or reference.is_complex()):
        return
    a = reference.detach().float().reshape(-1)
    difference = candidate.detach().float().reshape(-1) - a
    stats["reference_sq"] += float(torch.dot(a, a))
    stats["difference_sq"] += float(torch.dot(difference, difference))
    if difference.numel():
        stats["max_abs_delta"] = max(
            float(stats["max_abs_delta"]), float(difference.abs().max())
        )


def finish(stats: dict[str, float | int]) -> dict[str, float | int | None]:
    reference_l2 = math.sqrt(float(stats.pop("reference_sq")))
    difference_l2 = math.sqrt(float(stats.pop("difference_sq")))
    elements = int(stats["elements"])
    changed = int(stats["elements_changed"])
    stats["l2_reference"] = reference_l2
    stats["l2_difference"] = difference_l2
    stats["relative_l2"] = difference_l2 / reference_l2 if reference_l2 else None
    stats["changed_fraction"] = changed / elements if elements else None
    return stats


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-epoch", type=int, required=True)
    parser.add_argument("--candidate-epoch", type=int, required=True)
    parser.add_argument("--candidate-trainer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    before = {"reference": sha256(args.reference), "candidate": sha256(args.candidate)}
    reference = load(args.reference)
    candidate = load(args.candidate)
    if int(reference.get("current_epoch", -1)) != args.reference_epoch:
        raise ValueError("reference epoch mismatch")
    if int(candidate.get("current_epoch", -1)) != args.candidate_epoch:
        raise ValueError("candidate epoch mismatch")
    if candidate.get("trainer_name") != args.candidate_trainer:
        raise ValueError("candidate trainer mismatch")
    a_weights = reference["network_weights"]
    b_weights = candidate["network_weights"]
    if a_weights.keys() != b_weights.keys():
        raise ValueError("checkpoint tensor-key inventory mismatch")

    groups = defaultdict(blank)
    for key, a in a_weights.items():
        b = b_weights[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"checkpoint tensor contract mismatch: {key}")
        top = key.split(".", 1)[0]
        for group in ("all", top):
            update(groups[group], a, b)
    after = {"reference": sha256(args.reference), "candidate": sha256(args.candidate)}
    if before != after:
        raise RuntimeError("checkpoint changed during audit")

    payload = {
        "schema_version": 1,
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "status": "PASS",
        "purpose": "label-blind diagnostic of optimization magnitude; not a selection authority",
        "reference": {
            "path": str(args.reference.resolve()),
            "sha256": before["reference"],
            "trainer": reference.get("trainer_name"),
            "epoch": args.reference_epoch,
        },
        "candidate": {
            "path": str(args.candidate.resolve()),
            "sha256": before["candidate"],
            "trainer": candidate.get("trainer_name"),
            "epoch": args.candidate_epoch,
        },
        "groups": {key: finish(value) for key, value in sorted(groups.items())},
        "hash_stable_during_audit": True,
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
