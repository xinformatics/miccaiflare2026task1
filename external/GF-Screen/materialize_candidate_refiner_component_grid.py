#!/usr/bin/env python3
"""Run the shared grid materializer with hash-bound component confidences."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as real_numpy

import materialize_candidate_refiner_grid as shared
from component_meta_classifier import sha256


class OverrideArchive:
    def __init__(self, archive, probability: float):
        self.archive = archive
        self.probability = probability

    def __enter__(self):
        self.archive.__enter__()
        return self

    def __exit__(self, *args):
        return self.archive.__exit__(*args)

    def __getitem__(self, name):
        if name == "classification_probability":
            return real_numpy.asarray(self.probability, dtype=real_numpy.float32)
        return self.archive[name]

    def __getattr__(self, name):
        return getattr(self.archive, name)


class NumpyOverrideProxy:
    def __init__(self, work: Path, probabilities: dict[str, float]):
        self.work = work.resolve()
        self.probabilities = probabilities

    def __getattr__(self, name):
        return getattr(real_numpy, name)

    def load(self, file, *args, **kwargs):
        archive = real_numpy.load(file, *args, **kwargs)
        path = Path(file).resolve()
        if path.parent == self.work / "predictions" and path.stem in self.probabilities:
            return OverrideArchive(archive, self.probabilities[path.stem])
        return archive


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--configurations", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--segmentation-threshold", type=float, required=True)
    parser.add_argument("--component-probabilities", type=Path, required=True)
    parser.add_argument("--empty-base", action="store_true")
    parser.add_argument("--min-seed-coverage", type=float, default=0.05)
    parser.add_argument("--min-refined-overlap", type=float, default=0.05)
    parser.add_argument("--min-volume-ratio", type=float, default=0.25)
    parser.add_argument("--max-volume-ratio", type=float, default=4.0)
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--writer-workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    manifest = json.loads((args.work / "manifest.json").read_text(encoding="utf-8"))
    expected = {str(record["patch_id"]) for record in manifest.get("records", [])}
    with real_numpy.load(args.component_probabilities, allow_pickle=False) as archive:
        patch_ids = [str(value) for value in archive["patch_id"]]
        values = archive["classification_probability"].astype(real_numpy.float64)
    if (
        len(patch_ids) != len(set(patch_ids))
        or set(patch_ids) != expected
        or values.shape != (len(expected),)
        or not real_numpy.isfinite(values).all()
        or real_numpy.any((values < 0) | (values > 1))
    ):
        raise ValueError("component probability override lacks exact valid coverage")
    probabilities = dict(zip(patch_ids, values.tolist()))
    original_numpy = shared.np
    shared.np = NumpyOverrideProxy(args.work, probabilities)
    try:
        shared.materialize_grid(
            args.work,
            args.base,
            args.output_root,
            args.configurations,
            args.fold,
            args.segmentation_threshold,
            args.empty_base,
            args.min_seed_coverage,
            args.min_refined_overlap,
            args.min_volume_ratio,
            args.max_volume_ratio,
            args.case_limit,
            args.writer_workers,
        )
    finally:
        shared.np = original_numpy

    configuration_payload = json.loads(args.configurations.read_text(encoding="utf-8"))
    configurations = configuration_payload.get("configurations", configuration_payload)
    for configuration in configurations:
        marker = (
            args.output_root
            / str(configuration["tag"])
            / f"fold{args.fold}"
            / "CANDIDATE_REFINER_COMPLETE.json"
        )
        payload = json.loads(marker.read_text(encoding="utf-8"))
        payload.update(
            {
                "materializer": "shared_case_grid_component_override_v1",
                "component_probability_override": str(
                    args.component_probabilities.resolve()
                ),
                "component_probability_override_sha256": sha256(
                    args.component_probabilities
                ),
                "labels_read_by_override": False,
            }
        )
        atomic_json(marker, payload)


if __name__ == "__main__":
    main()
