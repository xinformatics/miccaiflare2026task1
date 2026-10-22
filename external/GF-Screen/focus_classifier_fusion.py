#!/usr/bin/env python3
"""Inference-only Focus/B32 confidence stacking with strict provenance.

The fitted coefficients are learned exclusively from grouped exact707 OOF
predictions.  This module only applies a previously qualified frozen model; it
does not fit, tune, or read labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from candidate_refiner_sampling import semantic_source_group


FOCUS_CONFIDENCE_FEATURE_NAMES = (
    "foreground_max",
    "foreground_top64_mean",
    "foreground_top512_mean",
    "foreground_fraction_0p50",
    "foreground_fraction_0p70",
    "foreground_fraction_0p90",
    "seed_probability_mean",
    "seed_probability_max",
    "prediction_0p50_seed_precision",
    "seed_coverage_0p50",
)
SOURCE_GROUP_NAMES = ("mature", "weak_sources", "lung_cells")
DEPLOYMENT_FEATURE_NAMES = (
    ("b32_logit",)
    + tuple(f"focus_{name}" for name in FOCUS_CONFIDENCE_FEATURE_NAMES)
    + ("log1p_seed_mm3", "max_source_support", "mean_source_support")
    + tuple(f"source_{name}" for name in SOURCE_GROUP_NAMES)
)


def focus_confidence_features(
    probabilities: torch.Tensor,
    images: torch.Tensor,
    input_mode: str,
) -> torch.Tensor:
    """Compute the exact inference-visible features used by the OOF stacker."""

    if probabilities.ndim != 5 or probabilities.shape[1] != 1:
        raise ValueError("Focus probabilities must have shape (B,1,D,H,W)")
    if images.ndim != 5 or len(images) != len(probabilities):
        raise ValueError("Focus images and probabilities must align")
    if input_mode not in {"soft", "soft_seed", "soft_lung_seed"}:
        raise ValueError(f"unsupported Focus input mode: {input_mode}")
    values = probabilities.float().flatten(1)
    top_count = min(512, values.shape[1])
    if top_count <= 0:
        raise ValueError("empty Focus probability volume")
    top = torch.topk(values, k=top_count, dim=1, sorted=True).values
    features = [
        top[:, 0],
        top[:, : min(64, top_count)].mean(1),
        top.mean(1),
        (values >= 0.50).float().mean(1),
        (values >= 0.70).float().mean(1),
        (values >= 0.90).float().mean(1),
    ]
    if input_mode in {"soft_seed", "soft_lung_seed"}:
        # Candidate-refiner caches always store seed as their final channel.
        if images.shape[1] < 2:
            raise ValueError("prompt-enabled Focus fusion requires a seed channel")
        seed = images[:, -1:] > 0.5
        seed_count = seed.flatten(1).sum(1).clamp_min(1)
        seed_values = probabilities.float().masked_fill(~seed, 0.0)
        seed_mean = seed_values.flatten(1).sum(1) / seed_count
        seed_max = probabilities.float().masked_fill(~seed, -1.0).flatten(1).max(1).values
        predicted = probabilities >= 0.50
        intersection = (predicted & seed).flatten(1).sum(1).float()
        features.extend(
            [
                seed_mean,
                seed_max.clamp_min(0.0),
                intersection / predicted.flatten(1).sum(1).clamp_min(1).float(),
                intersection / seed_count.float(),
            ]
        )
    else:
        zeros = torch.zeros_like(features[0])
        features.extend([zeros, zeros, zeros, zeros])
    result = torch.stack(features, dim=1)
    if result.shape[1] != len(FOCUS_CONFIDENCE_FEATURE_NAMES):
        raise RuntimeError("Focus confidence feature count mismatch")
    if not bool(torch.isfinite(result).all()):
        raise ValueError("non-finite Focus confidence feature")
    return result


def source_group(source_name: str) -> str:
    group = semantic_source_group(source_name)
    if group in {"epoch105", "currentgf"}:
        return "mature"
    if group in {"autopet_weak", "lndb_weak"}:
        return "weak_sources"
    if group != "lung_cells":
        raise ValueError(f"unsupported fusion source group: {source_name}")
    return group


def load_qualified_fusion(path: Path) -> dict[str, object]:
    """Load and fail closed on an unqualified or malformed OOF stacker."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("selection_basis")
        != "cross-fitted exact707 grouped held-out patch classification only"
        or payload.get("public_read") is not False
        or payload.get("hidden_read") is not False
        or payload.get("submission_created") is not False
    ):
        raise ValueError("Focus classifier fusion provenance is not local exact707 OOF")
    if payload.get("decision") != "QUALIFY_FOCUS_CLASSIFIER_FUSION":
        raise ValueError("Focus classifier fusion was not qualified on exact707 OOF")
    checks = payload.get("checks", {})
    if set(checks) != {
        "average_precision_gain",
        "f0p5_gain",
        "source_nonregression",
    } or not all(value is True for value in checks.values()):
        raise ValueError("Focus classifier fusion qualification checks are incomplete")
    model_identity = payload.get("model", {})
    if (
        model_identity.get("type") != "standardized logistic regression"
        or float(model_identity.get("C", np.nan)) != 0.1
    ):
        raise ValueError("Focus classifier fusion model identity mismatch")
    provenance = payload.get("provenance", [])
    if (
        not isinstance(provenance, list)
        or sorted(int(record.get("fold", -1)) for record in provenance)
        != list(range(5))
        or len(str(payload.get("index_sha256", ""))) != 64
        or any(
            len(str(record.get(name, ""))) != 64
            for record in provenance
            for name in ("b32_prediction_sha256", "focus_prediction_sha256")
        )
    ):
        raise ValueError("Focus classifier fusion lacks five-fold hash provenance")
    if tuple(payload.get("feature_names", ())) != DEPLOYMENT_FEATURE_NAMES:
        raise ValueError("Focus classifier fusion feature identity mismatch")
    input_mode = str(payload.get("focus_input_mode", ""))
    if input_mode not in {"soft", "soft_seed", "soft_lung_seed"}:
        raise ValueError("Focus classifier fusion lacks a valid frozen input mode")
    model = payload.get("deployment_model")
    if not isinstance(model, dict):
        raise ValueError("Focus classifier fusion lacks a deployment model")
    count = len(DEPLOYMENT_FEATURE_NAMES)
    arrays = {
        name: np.asarray(model.get(name), dtype=np.float64)
        for name in ("scaler_mean", "scaler_scale", "coefficient")
    }
    if any(values.shape != (count,) for values in arrays.values()):
        raise ValueError("Focus classifier fusion coefficient shape mismatch")
    intercept = float(model.get("intercept", np.nan))
    if (
        not all(np.isfinite(values).all() for values in arrays.values())
        or not np.isfinite(intercept)
        or np.any(arrays["scaler_scale"] <= 0)
    ):
        raise ValueError("Focus classifier fusion contains invalid coefficients")
    return {
        "focus_input_mode": input_mode,
        **arrays,
        "intercept": intercept,
    }


def apply_fusion(
    b32_probabilities: torch.Tensor,
    focus_probabilities: torch.Tensor,
    images: torch.Tensor,
    records: list[dict[str, object]],
    deployment: dict[str, object],
) -> torch.Tensor:
    """Apply a frozen standardized logistic stacker to one inference batch."""

    if b32_probabilities.ndim != 1 or len(b32_probabilities) != len(records):
        raise ValueError("B32 probabilities and inference records must align")
    mode = str(deployment["focus_input_mode"])
    focus = focus_confidence_features(focus_probabilities, images, mode)
    probability = b32_probabilities.float().clamp(1e-6, 1 - 1e-6)
    b32_logit = torch.log(probability / (1 - probability)).unsqueeze(1)
    geometry = torch.as_tensor(
        [
            [
                np.log1p(float(record.get("seed_mm3", 0.0) or 0.0)),
                float(record.get("max_source_support", 0.0) or 0.0),
                float(record.get("mean_source_support", 0.0) or 0.0),
            ]
            for record in records
        ],
        dtype=torch.float32,
        device=images.device,
    )
    one_hot = torch.as_tensor(
        [
            [float(source_group(str(record.get("source_name", ""))) == name) for name in SOURCE_GROUP_NAMES]
            for record in records
        ],
        dtype=torch.float32,
        device=images.device,
    )
    features = torch.cat((b32_logit, focus, geometry, one_hot), dim=1)
    if features.shape[1] != len(DEPLOYMENT_FEATURE_NAMES) or not bool(
        torch.isfinite(features).all()
    ):
        raise ValueError("invalid Focus classifier fusion features")
    mean = torch.as_tensor(deployment["scaler_mean"], dtype=torch.float32, device=images.device)
    scale = torch.as_tensor(deployment["scaler_scale"], dtype=torch.float32, device=images.device)
    coefficient = torch.as_tensor(deployment["coefficient"], dtype=torch.float32, device=images.device)
    intercept = torch.as_tensor(deployment["intercept"], dtype=torch.float32, device=images.device)
    logits = ((features - mean) / scale).matmul(coefficient) + intercept
    result = torch.sigmoid(logits)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("non-finite fused classification probability")
    return result
