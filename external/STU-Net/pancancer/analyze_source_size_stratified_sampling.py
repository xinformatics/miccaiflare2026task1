#!/usr/bin/env python3
"""Materialize the exact size-preserving source-stratified sampling audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from STUNetTrainerPanCancerSourceBalanced import (  # noqa: E402
    RAW_RATIO_CAP,
    SOURCE_FRACTION,
    foreground_proxy_counts_from_dataset,
    size_bin,
    source_from_identifier,
    stratified_source_probabilities,
)


class IdentifierDataset:
    def __init__(self, source_folder: Path, identifiers: list[str]):
        self.source_folder = str(source_folder)
        self.identifiers = identifiers


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocessed-folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-cases", type=int, default=17_575)
    parser.add_argument("--expected-sources", type=int, default=28)
    parser.add_argument("--draws-per-epoch", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    identifiers = sorted(path.stem for path in args.preprocessed_folder.glob("*.pkl"))
    if len(identifiers) != args.expected_cases:
        raise ValueError(
            f"expected {args.expected_cases} cases, found {len(identifiers)}"
        )
    sources = [source_from_identifier(identifier) for identifier in identifiers]
    if len(set(sources)) != args.expected_sources:
        raise ValueError(
            f"expected {args.expected_sources} sources, found {len(set(sources))}"
        )
    dataset = IdentifierDataset(args.preprocessed_folder, identifiers)
    foreground_counts = foreground_proxy_counts_from_dataset(dataset)
    probabilities = stratified_source_probabilities(
        identifiers,
        foreground_counts,
        SOURCE_FRACTION,
        RAW_RATIO_CAP,
    )
    uniform = np.full(len(identifiers), 1.0 / len(identifiers))
    bins = np.asarray([size_bin(int(count)) for count in foreground_counts])
    source_counts = Counter(sources)
    source_current = Counter()
    source_proposed = Counter()
    for source, current, proposed in zip(
        sources, uniform, probabilities, strict=True
    ):
        source_current[source] += float(current)
        source_proposed[source] += float(proposed)

    bin_rows = []
    for bin_index in sorted(set(int(value) for value in bins)):
        mask = bins == bin_index
        bin_rows.append(
            {
                "bin_index": bin_index,
                "case_count": int(mask.sum()),
                "current_probability_mass": float(uniform[mask].sum()),
                "proposed_probability_mass": float(probabilities[mask].sum()),
                "current_expected_draws_per_epoch": float(
                    args.draws_per_epoch * uniform[mask].sum()
                ),
                "proposed_expected_draws_per_epoch": float(
                    args.draws_per_epoch * probabilities[mask].sum()
                ),
            }
        )
    source_rows = []
    for source, count in sorted(
        source_counts.items(), key=lambda item: (-item[1], item[0])
    ):
        source_rows.append(
            {
                "source": source,
                "case_count": count,
                "current_probability_mass": source_current[source],
                "proposed_probability_mass": source_proposed[source],
                "mass_ratio": source_proposed[source] / source_current[source],
                "current_expected_draws_per_epoch": (
                    args.draws_per_epoch * source_current[source]
                ),
                "proposed_expected_draws_per_epoch": (
                    args.draws_per_epoch * source_proposed[source]
                ),
            }
        )

    payload = {
        "schema_version": 1,
        "status": "analysis_only_not_launched",
        "policy": "size_preserving_bounded_source_mixture",
        "source_fraction": SOURCE_FRACTION,
        "raw_per_case_ratio_cap": RAW_RATIO_CAP,
        "size_bin_upper_bounds": [100, 500, 1_000, 3_000, 9_999, "10000-capped"],
        "foreground_count_warning": (
            "nnU-Net class_locations count is a foreground-size proxy, may combine "
            "components, and is capped at 10000"
        ),
        "case_count": len(identifiers),
        "source_count": len(source_counts),
        "draws_per_epoch": args.draws_per_epoch,
        "probability_sum": float(probabilities.sum()),
        "effective_sample_size": float(1.0 / np.square(probabilities).sum()),
        "maximum_vs_uniform_case_weight": float(probabilities.max() * len(identifiers)),
        "minimum_vs_uniform_case_weight": float(probabilities.min() * len(identifiers)),
        "foreground_proxy_empty_cases": int((foreground_counts == 0).sum()),
        "foreground_proxy_capped_cases": int((foreground_counts >= 10_000).sum()),
        "bins": bin_rows,
        "sources": source_rows,
        "identifiers": [
            {
                "identifier": identifier,
                "source": source,
                "foreground_proxy_count": int(count),
                "size_bin": int(bin_index),
                "sampling_probability": float(probability),
            }
            for identifier, source, count, bin_index, probability in zip(
                identifiers,
                sources,
                foreground_counts,
                bins,
                probabilities,
                strict=True,
            )
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "status",
                    "case_count",
                    "source_count",
                    "source_fraction",
                    "raw_per_case_ratio_cap",
                    "effective_sample_size",
                    "maximum_vs_uniform_case_weight",
                    "minimum_vs_uniform_case_weight",
                )
            },
            indent=2,
        )
    )
    print(f"output={args.output}")
    print(f"sha256={sha256_file(args.output)}")


if __name__ == "__main__":
    main()
