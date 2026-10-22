#!/usr/bin/env python3
"""Leakage-safe component-level features and frozen classifier loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import joblib
import numpy as np


BASE_FEATURE_NAMES = (
    "patch_probability_max",
    "patch_probability_mean",
    "patch_probability_min",
    "patch_probability_std",
    "patch_probability_q25",
    "patch_probability_q75",
    "log1p_tile_count",
    "log1p_seed_mm3",
    "max_source_support",
    "mean_source_support",
)

# These are the exact proposal sources in the frozen exact-707 deployment
# inventory.  Explicit one-hot encoding prevents a model from silently
# accepting a new source with undefined calibration.
SOURCE_NAMES = (
    "autopet_weak",
    "currentgf",
    "epoch105",
    "lndb_weak",
    "lung_cell0",
    "lung_cell1",
    "lung_cell2",
    "lung_cell3",
    "lung_cell4",
    "lung_cell5",
    "lung_cell6",
    "lung_cell7",
)
FEATURE_NAMES = BASE_FEATURE_NAMES + tuple(f"source_{name}" for name in SOURCE_NAMES)

# These index columns directly use the held-out reference mask and must never
# enter a deployment feature vector.
FORBIDDEN_LABEL_FEATURES = frozenset(
    {
        "candidate_positive",
        "label",
        "intersection_voxels",
        "seed_overlap_fraction",
        "distance_to_target_mm",
        "target_resample_fallback",
    }
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _consistent_float(records: Sequence[Mapping[str, object]], name: str) -> float:
    values = np.asarray([float(record.get(name, 0.0) or 0.0) for record in records])
    if not np.isfinite(values).all() or not np.allclose(values, values[0], rtol=0, atol=1e-8):
        raise ValueError(f"component metadata {name!r} is inconsistent across tiles")
    return float(values[0])


def component_features(
    records: Sequence[Mapping[str, object]],
    probabilities: Iterable[float],
) -> np.ndarray:
    """Create one label-free feature vector for a source-separated component."""

    if not records:
        raise ValueError("a component must contain at least one tile")
    values = np.asarray(list(probabilities), dtype=np.float64)
    if values.shape != (len(records),) or not np.isfinite(values).all():
        raise ValueError("component probabilities must be finite and tile-aligned")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("component probabilities must lie in [0,1]")
    sources = {str(record.get("source_name", "")) for record in records}
    if len(sources) != 1:
        raise ValueError("component tiles must share one proposal source")
    source = next(iter(sources))
    if source not in SOURCE_NAMES:
        raise ValueError(f"unexpected proposal source: {source!r}")
    seed_mm3 = _consistent_float(records, "seed_mm3")
    max_support = _consistent_float(records, "max_source_support")
    mean_support = _consistent_float(records, "mean_source_support")
    if seed_mm3 < 0 or not 0 <= mean_support <= max_support <= 1:
        raise ValueError("invalid component geometry/support metadata")
    features = np.asarray(
        [
            float(values.max()),
            float(values.mean()),
            float(values.min()),
            float(values.std()),
            float(np.quantile(values, 0.25)),
            float(np.quantile(values, 0.75)),
            float(np.log1p(len(values))),
            float(np.log1p(seed_mm3)),
            max_support,
            mean_support,
            *[float(source == name) for name in SOURCE_NAMES],
        ],
        dtype=np.float64,
    )
    if features.shape != (len(FEATURE_NAMES),) or not np.isfinite(features).all():
        raise RuntimeError("invalid component feature vector")
    return features


def load_qualified_component_classifier(
    decision_path: Path,
    fold: int | None = None,
) -> tuple[object, dict[str, object]]:
    """Load a hash-bound exact-707 model, failing closed on provenance drift."""

    decision_path = decision_path.resolve()
    payload = json.loads(decision_path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("decision") != "QUALIFY_COMPONENT_META_CLASSIFIER"
        or payload.get("selection_basis")
        != "cross-fitted exact707 grouped held-out component classification only"
        or payload.get("public_read") is not False
        or payload.get("hidden_read") is not False
        or payload.get("submission_created") is not False
    ):
        raise ValueError("component classifier is not a qualified local-only model")
    if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("component classifier feature identity drift")
    model_path = Path(str(payload.get("model_bundle", ""))).resolve()
    if not model_path.is_file() or sha256(model_path) != payload.get("model_bundle_sha256"):
        raise ValueError("component classifier model identity mismatch")
    bundle = joblib.load(model_path)
    if tuple(bundle.get("feature_names", ())) != FEATURE_NAMES:
        raise ValueError("component classifier bundle feature mismatch")
    if tuple(bundle.get("source_names", ())) != SOURCE_NAMES:
        raise ValueError("component classifier bundle source mismatch")
    if fold is None:
        model = bundle.get("deployment_model")
    else:
        models = bundle.get("fold_models", {})
        model = models.get(int(fold))
    if model is None or not hasattr(model, "predict_proba"):
        raise ValueError(f"component classifier lacks requested model fold={fold}")
    return model, payload


def predict_component_probability(
    model: object,
    records: Sequence[Mapping[str, object]],
    probabilities: Iterable[float],
) -> float:
    features = component_features(records, probabilities)
    prediction = np.asarray(model.predict_proba(features[None]))
    if prediction.shape != (1, 2) or not np.isfinite(prediction).all():
        raise ValueError("component classifier returned invalid probabilities")
    value = float(prediction[0, 1])
    if not 0 <= value <= 1:
        raise ValueError("component probability lies outside [0,1]")
    return value
