#!/usr/bin/env python3
"""Rank saved candidates with uncertainty from paired public/hidden history.

The calibration is diagnostic, not a hidden-score oracle.  It deliberately
uses one public metric at a time because leave-one-out testing shows that
adding the other public metric does not improve prediction error.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


def fit_interval(x: np.ndarray, y: np.ndarray, query: float) -> dict[str, float]:
    design = np.column_stack((np.ones(len(x)), x))
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ coefficients
    residual = y - fitted
    dof = len(x) - 2
    residual_std = float(np.sqrt(np.sum(residual**2) / dof))
    mean = float(coefficients[0] + coefficients[1] * query)
    leverage = 1.0 / len(x) + (query - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2)
    half95 = float(t.ppf(0.975, dof) * residual_std * np.sqrt(1.0 + leverage))
    return {
        "mean": mean,
        "prediction_low95": mean - half95,
        "prediction_high95": mean + half95,
        "slope": float(coefficients[1]),
        "intercept": float(coefficients[0]),
        "residual_std": residual_std,
    }


def loocv(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    predictions = []
    for held_out in range(len(x)):
        keep = np.arange(len(x)) != held_out
        design = np.column_stack((np.ones(keep.sum()), x[keep]))
        coefficients = np.linalg.lstsq(design, y[keep], rcond=None)[0]
        predictions.append(coefficients[0] + coefficients[1] * x[held_out])
    error = np.asarray(predictions) - y
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "maximum_absolute_error": float(np.max(np.abs(error))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    args = parser.parse_args()

    with args.history.open(newline="", encoding="utf-8") as handle:
        history = [
            row
            for row in csv.DictReader(handle, delimiter="\t")
            if all(
                row.get(key, "").strip()
                for key in (
                    "public_dsc",
                    "public_nsd",
                    "codabench_dsc",
                    "codabench_nsd",
                )
            )
        ]
    with args.candidates.open(newline="", encoding="utf-8") as handle:
        candidates = list(csv.DictReader(handle, delimiter="\t"))
    if len(history) < 5:
        raise ValueError("at least five complete public/hidden pairs are required")

    public_dsc = np.array([float(row["public_dsc"]) for row in history])
    hidden_dsc = np.array([float(row["codabench_dsc"]) for row in history])
    public_nsd = np.array([float(row["public_nsd"]) for row in history])
    hidden_nsd = np.array([float(row["codabench_nsd"]) for row in history])
    diagnostics = {
        "paired_history_count": len(history),
        "ranking_objective": (
            "predicted hidden DSC first; predicted hidden NSD tie-breaker "
            "after the upstream strict two-metric promotion gate"
        ),
        "warning": (
            "Prediction intervals are broad and model-family shifts are not "
            "represented. Use this only to prioritize, never to claim a score."
        ),
        "dsc_loocv": loocv(public_dsc, hidden_dsc),
        "nsd_loocv": loocv(public_nsd, hidden_nsd),
    }

    ranked = []
    for row in candidates:
        dsc = fit_interval(public_dsc, hidden_dsc, float(row["DSC"]))
        nsd = fit_interval(public_nsd, hidden_nsd, float(row["NSD"]))
        ranked.append(
            {
                **row,
                "predicted_hidden_DSC": dsc["mean"],
                "DSC_low95": dsc["prediction_low95"],
                "DSC_high95": dsc["prediction_high95"],
                "predicted_hidden_NSD": nsd["mean"],
                "NSD_low95": nsd["prediction_low95"],
                "NSD_high95": nsd["prediction_high95"],
                "predicted_hidden_combined": (dsc["mean"] + nsd["mean"]) / 2,
            }
        )
    ranked.sort(
        key=lambda row: (
            row["predicted_hidden_DSC"],
            row["predicted_hidden_NSD"],
            row["predicted_hidden_combined"],
        ),
        reverse=True,
    )
    for index, row in enumerate(ranked, 1):
        row["calibrated_rank"] = index

    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "calibrated_rank",
        "tag",
        "family",
        "DSC",
        "NSD",
        "predicted_hidden_DSC",
        "DSC_low95",
        "DSC_high95",
        "predicted_hidden_NSD",
        "NSD_low95",
        "NSD_high95",
        "predicted_hidden_combined",
        "submission",
        "sha256",
    )
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for row in ranked:
            writer.writerow({column: row.get(column, "") for column in columns})
    args.json_output.write_text(
        json.dumps({"diagnostics": diagnostics, "ranked": ranked}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(diagnostics, indent=2))
    for row in ranked:
        print(
            f'{row["calibrated_rank"]}: {row["tag"]} '
            f'DSC={row["predicted_hidden_DSC"]:.4f} '
            f'[{row["DSC_low95"]:.4f}, {row["DSC_high95"]:.4f}] '
            f'NSD={row["predicted_hidden_NSD"]:.4f} '
            f'[{row["NSD_low95"]:.4f}, {row["NSD_high95"]:.4f}]'
        )


if __name__ == "__main__":
    main()
