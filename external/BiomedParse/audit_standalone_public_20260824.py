#!/usr/bin/env python3
"""Audit every deliberately standalone BiomedParse result on the 50 public cases.

This excludes guided/fused predictions, which contain GF-Screen masks and do
not answer the question "how does BiomedParse perform by itself?".  The report
is descriptive only: public labels are read, no hidden prediction is produced,
and no submission is authorized.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = ("Lesion_DSC", "Lesion_NSD")
CURATED_PATTERNS = (
    "automatic_text_proposals_20260820/scored/standalone_*.metrics.csv",
    "organ_specific_rescue_20260821/standalone_metrics/*.metrics.csv",
    "standalone_malignant_filter_20260821/metrics/*.metrics.csv",
    "standalone_malignant_filter_extension_20260821/metrics/*.metrics.csv",
    "standalone_prompt_banks_20260821/metrics/*.metrics.csv",
)


def case_name(path: Path) -> str:
    name = path.name
    return name[:-7] if name.endswith(".nii.gz") else path.stem


def expected_cases(labels_dir: Path) -> set[str]:
    names = {case_name(path) for path in labels_dir.glob("*.nii.gz")}
    if not names:
        raise ValueError(f"no .nii.gz labels found in {labels_dir}")
    return names


def read_metrics(path: Path, expected: set[str]) -> dict[str, np.ndarray | list[str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"no metric rows in {path}")
    required = {"Name", *METRICS}
    missing_columns = required - set(rows[0])
    if missing_columns:
        raise ValueError(f"missing columns {sorted(missing_columns)} in {path}")
    names = [case_name(Path(row["Name"])) for row in rows]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate cases in {path}")
    observed = set(names)
    if observed != expected:
        raise ValueError(
            f"case mismatch in {path}: missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )
    order = np.argsort(np.asarray(names))
    ordered_names = [names[index] for index in order]
    output: dict[str, np.ndarray | list[str]] = {"names": ordered_names}
    for metric in METRICS:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)[order]
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError(f"invalid {metric} values in {path}")
        output[metric] = values
    return output


def discover(root: Path) -> list[Path]:
    paths = {path.resolve() for pattern in CURATED_PATTERNS for path in root.glob(pattern)}
    if not paths:
        raise ValueError(f"no curated standalone metrics found below {root}")
    return sorted(paths)


def bootstrap_delta_ci(
    candidate: np.ndarray,
    anchor: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> list[float]:
    delta = candidate - anchor
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 2000):
        stop = min(samples, start + 2000)
        indices = rng.integers(0, len(delta), size=(stop - start, len(delta)))
        means[start:stop] = delta[indices].mean(axis=1)
    low, high = np.percentile(means, (2.5, 97.5))
    return [float(low), float(high)]


def summarize(
    candidates: list[Path],
    labels_dir: Path,
    anchor_path: Path,
    *,
    bootstrap: int,
    seed: int,
) -> dict:
    expected = expected_cases(labels_dir)
    anchor = read_metrics(anchor_path, expected)
    anchor_matrix = np.column_stack([anchor[metric] for metric in METRICS])
    ranking = []
    for rank_seed, path in enumerate(candidates):
        data = read_metrics(path, expected)
        if data["names"] != anchor["names"]:
            raise AssertionError("sorted case order mismatch")
        matrix = np.column_stack([data[metric] for metric in METRICS])
        delta = matrix - anchor_matrix
        combined_case_delta = delta.mean(axis=1)
        try:
            relative = path.resolve().relative_to(
                Path(__file__).resolve().parent / "public_sweeps"
            )
        except ValueError:
            relative = path.name
        ranking.append(
            {
                "name": str(relative).removesuffix(".metrics.csv"),
                "metrics_csv": str(path.resolve()),
                "cases": len(expected),
                "mean": {metric: float(matrix[:, i].mean()) for i, metric in enumerate(METRICS)},
                "combined_mean": float(matrix.mean()),
                "delta_vs_public_champion": {
                    metric: float(delta[:, i].mean()) for i, metric in enumerate(METRICS)
                },
                "combined_delta_vs_public_champion": float(combined_case_delta.mean()),
                "combined_delta_bootstrap_95ci": bootstrap_delta_ci(
                    matrix.mean(axis=1),
                    anchor_matrix.mean(axis=1),
                    samples=bootstrap,
                    seed=seed + rank_seed,
                ),
                "paired_win_rate_vs_public_champion": float((combined_case_delta > 0).mean()),
                "paired_tie_rate_vs_public_champion": float((combined_case_delta == 0).mean()),
            }
        )
    ranking.sort(key=lambda row: (row["combined_mean"], row["mean"]["Lesion_DSC"]), reverse=True)
    for index, row in enumerate(ranking, start=1):
        row["rank"] = index
    return {
        "question": "How does a BiomedParse-only model perform?",
        "scope": "curated standalone BiomedParse predictions; no GF-Screen guidance or fusion",
        "public_label_read": True,
        "hidden_label_read": False,
        "submission_generated": False,
        "cases": len(expected),
        "candidate_count": len(ranking),
        "anchor": {
            "name": "gf_champion_public_reference",
            "metrics_csv": str(anchor_path.resolve()),
            "mean": {
                metric: float(np.asarray(anchor[metric]).mean()) for metric in METRICS
            },
            "combined_mean": float(anchor_matrix.mean()),
        },
        "best": ranking[0],
        "decision": "KEEP_BIOMEDPARSE_AS_PROPOSAL_SOURCE_ONLY",
        "reason": (
            "The strongest standalone variant is substantially below the GF-Screen public "
            "anchor; retain BiomedParse for complementary proposal recall, not as a standalone submission."
        ),
        "ranking": ranking,
    }


def write_tsv(path: Path, rows: list[dict]) -> None:
    fields = (
        "rank",
        "name",
        "cases",
        "Lesion_DSC",
        "Lesion_NSD",
        "combined_mean",
        "DSC_delta_vs_anchor",
        "NSD_delta_vs_anchor",
        "combined_delta_vs_anchor",
        "paired_win_rate_vs_anchor",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "rank": row["rank"],
                    "name": row["name"],
                    "cases": row["cases"],
                    "Lesion_DSC": f'{row["mean"]["Lesion_DSC"]:.6f}',
                    "Lesion_NSD": f'{row["mean"]["Lesion_NSD"]:.6f}',
                    "combined_mean": f'{row["combined_mean"]:.6f}',
                    "DSC_delta_vs_anchor": f'{row["delta_vs_public_champion"]["Lesion_DSC"]:.6f}',
                    "NSD_delta_vs_anchor": f'{row["delta_vs_public_champion"]["Lesion_NSD"]:.6f}',
                    "combined_delta_vs_anchor": f'{row["combined_delta_vs_public_champion"]:.6f}',
                    "paired_win_rate_vs_anchor": f'{row["paired_win_rate_vs_public_champion"]:.6f}',
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--public-sweeps-root",
        type=Path,
        default=Path(__file__).resolve().parent / "public_sweeps",
    )
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    if args.bootstrap <= 0:
        raise ValueError("--bootstrap must be positive")
    report = summarize(
        discover(args.public_sweeps_root),
        args.labels_dir,
        args.anchor,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_tsv(args.output_dir / "ranking.tsv", report["ranking"])
    print(json.dumps({"report": str(report_path), "best": report["best"], "decision": report["decision"]}, indent=2))


if __name__ == "__main__":
    main()
