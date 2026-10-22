#!/usr/bin/env python3
"""Run candidate-refiner inference with x/y flip-TTA for GF Focus contours.

This deployment entry point intentionally leaves ``apply_candidate_refiner.py``
unchanged.  Existing exact-707 caches bind that implementation by SHA256, so
editing it while the primary grid is active would make a restarted worker
discard valid predictions.  The wrapper has its own inference identity and
applies TTA only to Focus contour models; candidate confidence and any routed
candidate-refiner contours remain bit-for-bit on the established path.

The wrapper is deliberately axis locked.  A different TTA recipe must use a
different entry point, preventing x, xy, and xyz predictions from sharing an
inference fingerprint accidentally.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import torch

import apply_candidate_refiner as deployment
from focus_candidate_refiner import foreground_logit


TTA_AXES = "xy"
TTA_DIMENSIONS = (2, 3)
EXPECTED_DEPLOYMENT_SHA256 = (
    "2c4cdeb242a1458aabeb0c389a794f47d4d751e11eb8bc65f6471d639036fde9"
)
TTA_MARKER_NAME = "FOCUS_TTA_XY_INFERENCE_COMPLETE.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class FocusXYTTA(torch.nn.Module):
    """Average original, x-flipped, and y-flipped Focus probabilities."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.candidate_refiner_input_mode = getattr(
            model, "candidate_refiner_input_mode", "soft"
        )

    def _probability(self, images: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(foreground_logit(self.model(images))).float()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        probability = self._probability(images)
        for dimension in TTA_DIMENSIONS:
            flipped = torch.flip(images, dims=(dimension,))
            restored = torch.flip(
                self._probability(flipped), dims=(dimension,)
            )
            probability = probability + restored
        probability = probability / (1 + len(TTA_DIMENSIONS))
        foreground = torch.logit(probability.clamp(1e-6, 1.0 - 1e-6))
        return torch.cat((torch.zeros_like(foreground), foreground), dim=1)


def tta_loader(
    loader: Callable[[Path, torch.device], torch.nn.Module],
    tta_paths: set[Path] | None = None,
) -> Callable[[Path, torch.device], torch.nn.Module]:
    """Wrap selected models loaded through the GF Focus contour loader."""

    selected = {path.resolve() for path in tta_paths} if tta_paths is not None else None

    def load(path: Path, device: torch.device) -> torch.nn.Module:
        model = loader(path, device)
        if selected is not None and path.resolve() not in selected:
            return model
        return FocusXYTTA(model).eval()

    return load


def parse_tta_targets(arguments: Sequence[str]) -> tuple[set[Path], list[str]]:
    """Resolve global or route-suffixed Focus checkpoints that receive TTA."""

    globals_: list[Path] = []
    routed: list[tuple[str, Path]] = []
    for index, value in enumerate(arguments):
        if value == "--contour-checkpoint":
            if index + 1 >= len(arguments):
                raise ValueError("missing --contour-checkpoint value")
            globals_.append(Path(arguments[index + 1]).resolve())
        elif value == "--focus-contour-route":
            if index + 1 >= len(arguments) or "=" not in arguments[index + 1]:
                raise ValueError("invalid --focus-contour-route value")
            name, raw_path = arguments[index + 1].split("=", 1)
            routed.append((name.strip(), Path(raw_path.strip()).resolve()))
    if globals_ and routed:
        raise ValueError("global and routed Focus contours are mutually exclusive")
    if globals_:
        return set(globals_), ["global"]
    if not routed:
        raise ValueError("xy-TTA deployment requires a GF Focus contour route")
    tta_routes = sorted(name for name, _ in routed if name.endswith("_tta_xy"))
    if not tta_routes:
        raise ValueError("routed xy-TTA requires at least one *_tta_xy route")
    tta_paths = {path for name, path in routed if name.endswith("_tta_xy")}
    notta_paths = {path for name, path in routed if not name.endswith("_tta_xy")}
    overlap = tta_paths & notta_paths
    if overlap:
        raise ValueError(
            "one resolved checkpoint cannot be both TTA and no-TTA; provide a "
            f"distinct hardlink for the TTA route: {sorted(map(str, overlap))}"
        )
    return tta_paths, tta_routes


def configure(tta_paths: set[Path]) -> None:
    implementation = Path(deployment.__file__).resolve()
    observed = sha256(implementation)
    if observed != EXPECTED_DEPLOYMENT_SHA256:
        raise RuntimeError(
            "apply_candidate_refiner.py changed after the Focus xy-TTA wrapper "
            f"was qualified: expected {EXPECTED_DEPLOYMENT_SHA256}, got {observed}"
        )
    deployment.load_focus_contour_model = tta_loader(
        deployment.load_focus_contour_model, tta_paths
    )
    # ``infer`` hashes ``Path(__file__)`` as part of its cache identity. Point
    # that identity at this axis-locked entry point so no-TTA and xy-TTA caches
    # can never be considered interchangeable.
    deployment.__file__ = str(Path(__file__).resolve())


def write_tta_marker(work: Path, tta_routes: list[str]) -> dict[str, object]:
    """Bind the TTA recipe to the completed base inference cache."""

    plan_path = work / "INFERENCE_PLAN.json"
    inference_path = work / "INFERENCE_COMPLETE.json"
    if not plan_path.is_file() or not inference_path.is_file():
        raise FileNotFoundError("TTA inference did not produce complete cache markers")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    inference = json.loads(inference_path.read_text(encoding="utf-8"))
    fingerprint = str(plan.get("inference_fingerprint", ""))
    if not fingerprint or inference.get("inference_fingerprint") != fingerprint:
        raise ValueError("TTA inference plan/complete fingerprint mismatch")
    implementation = plan.get("identity", {}).get("implementation", [])
    wrapper = Path(__file__).resolve()
    if (
        not isinstance(implementation, list)
        or not implementation
        or implementation[0].get("path") != str(wrapper)
        or implementation[0].get("sha256") != sha256(wrapper)
    ):
        raise ValueError("TTA inference plan is not bound to the xy wrapper")
    payload: dict[str, object] = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "tta_axes": TTA_AXES,
        "tta_views": 1 + len(TTA_DIMENSIONS),
        "focus_only": True,
        "tta_focus_routes": tta_routes,
        "inference_fingerprint": fingerprint,
        "inference_plan": str(plan_path.resolve()),
        "inference_plan_sha256": sha256(plan_path),
        "inference_complete": str(inference_path.resolve()),
        "inference_complete_sha256": sha256(inference_path),
        "implementation": str(wrapper),
        "implementation_sha256": sha256(wrapper),
        "underlying_deployment": str(
            wrapper.with_name("apply_candidate_refiner.py")
        ),
        "underlying_deployment_sha256": EXPECTED_DEPLOYMENT_SHA256,
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
    }
    marker = work / TTA_MARKER_NAME
    temporary = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, marker)
    return payload


def main() -> None:
    tta_paths, tta_routes = parse_tta_targets(sys.argv[1:])
    configure(tta_paths)
    deployment.main()
    if len(sys.argv) > 1 and sys.argv[1] == "infer":
        try:
            work = Path(sys.argv[sys.argv.index("--work") + 1])
        except (ValueError, IndexError) as error:
            raise ValueError("xy-TTA inference requires --work") from error
        write_tta_marker(work, tta_routes)


if __name__ == "__main__":
    main()
