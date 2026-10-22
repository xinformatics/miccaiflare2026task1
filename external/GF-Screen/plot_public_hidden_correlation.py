#!/usr/bin/env python3
"""Plot paired public-validation and Codabench hidden metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, default=Path("codabench_history.tsv"))
    parser.add_argument("--output", type=Path, default=Path("analysis/public_hidden_correlation.png"))
    parser.add_argument("--stats", type=Path, default=Path("analysis/public_hidden_correlation.json"))
    return parser.parse_args()


def short_label(row: dict[str, str]) -> str:
    model = row["model"]
    config = row["configuration"]
    replacements = {
        "wu_epoch105": "E105",
        "wu_full17k_ckpt150": "E150",
        "wise_focus_epoch105_to_215_a010": "WiSE .10",
        "wise_focus_epoch105_to_275_a015": "WiSE .15",
        "wise_focus_epoch105_to_275_a020": "WiSE .20",
        "wise_focus_epoch105_to_275_a0225": "WiSE .225",
        "wise_focus_epoch105_to_275_a025": "WiSE .25",
        "wise_focus_epoch105_to_275_a040": "WiSE .40",
        "wise_focus_epoch105_to_275_a055": "WiSE .55",
        "wise_focus_epoch105_to_275_a069": "WiSE .69",
        "wise_focus_epoch105_to_275_a100": "WiSE 1.0",
        "a055_bbox_pu_ep280": "BBox champion",
        "a055_bbox_ep280_beta0875": "BBox beta .875",
        "a055_bbox_ep280_beta1125": "BBox beta 1.125",
        "epoch105_ckpt150_ensemble": "E105/E150 ens.",
        "gf_bbox_component16_ep283": "Component16 E283",
        "gf_prompt_or_e050_norescue": "BBox + prompt OR",
        "gf_prompt_or_e050_focus075_tumor_mean_e010": "BBox + prompt rescue",
        "gf_v7a_e285_l0p50": "Verified7 appearance E285",
    }
    label = replacements.get(model, model)
    if "mm" in config:
        label += f" {config.rsplit('mm', 1)[-1]}mm³"
    return label


def main() -> None:
    args = parse_args()
    with args.history.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.6))
    fig.subplots_adjust(left=0.08, right=0.985, top=0.86, bottom=0.30, wspace=0.27)
    stats: dict[str, object] = {}
    legend_rows: list[dict[str, str]] | None = None

    specs = (
        ("DSC", "public_dsc", "codabench_dsc", "#1677b8"),
        ("NSD", "public_nsd", "codabench_nsd", "#d95f02"),
    )
    for ax, (metric, public_key, hidden_key, color) in zip(axes, specs):
        paired = [row for row in rows if row.get(public_key, "").strip()
                  and row.get(hidden_key, "").strip()]
        if legend_rows is None:
            legend_rows = paired
        if len(paired) < 3:
            raise ValueError(f"need at least 3 {metric} pairs, found {len(paired)}")
        x = np.array([float(row[public_key]) for row in paired])
        y = np.array([float(row[hidden_key]) for row in paired])
        slope, intercept = np.polyfit(x, y, 1)
        pearson = pearsonr(x, y)
        spearman = spearmanr(x, y)
        grid = np.linspace(x.min() - 0.005, x.max() + 0.005, 200)
        prediction = slope * x + intercept
        r_squared = 1.0 - np.sum((y - prediction) ** 2) / np.sum((y - y.mean()) ** 2)
        loo_prediction = np.empty_like(y)
        for held_out in range(len(y)):
            keep = np.arange(len(y)) != held_out
            loo_slope, loo_intercept = np.polyfit(x[keep], y[keep], 1)
            loo_prediction[held_out] = loo_slope * x[held_out] + loo_intercept
        loo_absolute_error = np.abs(y - loo_prediction)
        hidden_targets = (0.75, 0.80) if metric == "DSC" else (0.6369, 0.65)

        ax.scatter(x, y, s=65, color=color, edgecolor="white", linewidth=0.8, zorder=3)
        ax.plot(grid, slope * grid + intercept, color=color, linewidth=1.8, alpha=0.8)
        for index, row in enumerate(paired):
            ax.annotate(
                str(index + 1), (x[index], y[index]), xytext=(5, 4),
                textcoords="offset points", fontsize=7, weight="bold",
            )
        ax.set_title(f"Public vs hidden {metric}")
        ax.set_xlabel(f"Public validation {metric}")
        ax.set_ylabel(f"Codabench hidden {metric}")
        ax.grid(alpha=0.25)
        ax.text(
            0.03, 0.97,
            f"n={len(x)}\nPearson r={pearson.statistic:.3f} (p={pearson.pvalue:.3f})\n"
            f"Spearman ρ={spearman.statistic:.3f} (p={spearman.pvalue:.3f})\nR²={r_squared:.3f}",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
        )
        stats[metric.lower()] = {
            "n": len(paired),
            "slope": float(slope),
            "intercept": float(intercept),
            "pearson_r": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "spearman_rho": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
            "r_squared": float(r_squared),
            "loo_mae": float(loo_absolute_error.mean()),
            "loo_max_absolute_error": float(loo_absolute_error.max()),
            "inverse_targets": {
                f"hidden_{target:.4f}": {
                    "public_point_estimate": float((target - intercept) / slope),
                    "warning": (
                        "OLS inverse is a prioritization estimate, not a performance guarantee"
                    ),
                }
                for target in hidden_targets
            },
            "points": [
                {
                    "label": short_label(row),
                    "model": row["model"],
                    "configuration": row["configuration"],
                    "public": float(row[public_key]),
                    "hidden": float(row[hidden_key]),
                }
                for row in paired
            ],
        }

    fig.suptitle("GF-Screen: local public validation vs Codabench hidden validation", fontsize=14)
    assert legend_rows is not None
    handles = [
        Line2D([], [], marker="o", linestyle="none", color="0.35", markersize=4)
        for _ in legend_rows
    ]
    fig.legend(
        handles,
        [f"{index + 1}  {short_label(row)}" for index, row in enumerate(legend_rows)],
        loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=3,
        fontsize=7.5, frameon=False, handlelength=0.8, columnspacing=1.3,
    )
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    args.stats.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"wrote {args.output}")
    print(f"wrote {args.stats}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
