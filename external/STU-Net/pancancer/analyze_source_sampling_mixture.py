#!/usr/bin/env python3
"""Audit case-source imbalance and a bounded source-balanced sampling mixture.

This script is deliberately analysis-only. It reads the identifiers in an
nnU-Net preprocessed folder and writes probabilities that a future trainer may
pass to ``nnUNetDataLoader``. It does not import or modify the active trainer.

For case ``i`` belonging to source ``g``, the proposed probability is

    p_i = (1 - source_fraction) / N + source_fraction / (G * n_g)

where N is the case count, G the source count, and n_g the size of source g.
The mixture keeps substantial case-proportional sampling while bounding the
underexposure of small cohorts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def source_from_identifier(identifier: str) -> str:
    """Return the dataset source encoded before the final case token."""

    if "_" not in identifier:
        raise ValueError(f"identifier has no source separator: {identifier!r}")
    source, case_token = identifier.rsplit("_", 1)
    if not source or not case_token:
        raise ValueError(f"invalid identifier: {identifier!r}")
    return source


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_audit(
    identifiers: list[str], source_fraction: float, draws_per_epoch: int
) -> dict:
    if not identifiers:
        raise ValueError("no preprocessed case identifiers found")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("preprocessed identifiers are not unique")
    if not 0.0 <= source_fraction <= 1.0:
        raise ValueError("source_fraction must lie in [0, 1]")
    if draws_per_epoch <= 0:
        raise ValueError("draws_per_epoch must be positive")

    sources = [source_from_identifier(identifier) for identifier in identifiers]
    counts = Counter(sources)
    case_count = len(identifiers)
    source_count = len(counts)
    current = np.full(case_count, 1.0 / case_count, dtype=np.float64)
    proposed = np.asarray(
        [
            (1.0 - source_fraction) / case_count
            + source_fraction / (source_count * counts[source])
            for source in sources
        ],
        dtype=np.float64,
    )
    if not np.isclose(proposed.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise AssertionError(f"sampling probabilities sum to {proposed.sum()}")

    rows = []
    for source, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        current_mass = count / case_count
        proposed_mass = (1.0 - source_fraction) * current_mass + (
            source_fraction / source_count
        )
        rows.append(
            {
                "source": source,
                "case_count": count,
                "case_fraction": current_mass,
                "current_expected_draws_per_epoch": draws_per_epoch * current_mass,
                "proposed_probability_mass": proposed_mass,
                "proposed_expected_draws_per_epoch": draws_per_epoch * proposed_mass,
                "per_case_weight_ratio": proposed_mass / current_mass,
            }
        )

    return {
        "schema_version": 1,
        "status": "analysis_only_not_launched",
        "formula": "(1-source_fraction)/N + source_fraction/(G*n_source)",
        "source_fraction": source_fraction,
        "draws_per_epoch": draws_per_epoch,
        "case_count": case_count,
        "source_count": source_count,
        "current_effective_sample_size": float(1.0 / np.square(current).sum()),
        "proposed_effective_sample_size": float(1.0 / np.square(proposed).sum()),
        "probability_sum": float(proposed.sum()),
        "minimum_case_probability": float(proposed.min()),
        "maximum_case_probability": float(proposed.max()),
        "maximum_to_minimum_case_probability_ratio": float(
            proposed.max() / proposed.min()
        ),
        "sources": rows,
        "identifiers": [
            {
                "identifier": identifier,
                "source": source,
                "sampling_probability": float(probability),
            }
            for identifier, source, probability in zip(
                identifiers, sources, proposed, strict=True
            )
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed-folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-fraction", type=float, default=0.5)
    parser.add_argument("--draws-per-epoch", type=int, default=500)
    parser.add_argument("--expected-cases", type=int, default=17_575)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    identifiers = sorted(path.stem for path in args.preprocessed_folder.glob("*.pkl"))
    if len(identifiers) != args.expected_cases:
        raise ValueError(
            f"expected {args.expected_cases} cases, found {len(identifiers)} in "
            f"{args.preprocessed_folder}"
        )
    audit = build_audit(identifiers, args.source_fraction, args.draws_per_epoch)
    audit["preprocessed_folder"] = str(args.preprocessed_folder.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({key: audit[key] for key in (
        "status",
        "case_count",
        "source_count",
        "source_fraction",
        "current_effective_sample_size",
        "proposed_effective_sample_size",
        "maximum_to_minimum_case_probability_ratio",
    )}, indent=2))
    print(f"output={args.output}")
    print(f"sha256={sha256_file(args.output)}")


if __name__ == "__main__":
    main()
