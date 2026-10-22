#!/usr/bin/env python3
"""Gate five-fold encoder-adapted Focus routes against frozen Focus."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path

from summarize_focus_epoch_routes import sha256


ROUTES = ("weak_sources", "lung_cells")


def finite(values: dict[str, float]) -> None:
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("non-finite five-fold route metric")


def validate_focus(payload: dict, role: str, index_sha: str) -> None:
    if (
        int(payload.get("folds", -1)) != 5
        or int(payload.get("seed", -1)) != 20260823
        or int(payload.get("expected_epochs", -1)) != 11
        or payload.get("index_sha256") != index_sha
        or payload.get("public_read")
        or payload.get("hidden_read")
        or payload.get("submission_created")
        or set(payload.get("source_segmentation_groups", {}))
        != {"mature", "weak_sources", "lung_cells"}
    ):
        raise ValueError(f"{role} is not matched exact five-fold routed Focus")
    recipe = payload.get("training_recipe", {})
    expected = {
        "epochs": 10,
        "batch_size": 8,
        "lr": 3e-5,
        "weight_decay": 1e-5,
        "loss_recipe": "release_dice_ce",
        "sampling_temperature": 0.5,
        "samples_per_epoch": 2048,
        "input_mode": "soft_lung_seed",
        "segmentation_threshold_max": 0.95,
    }
    mismatches = {
        key: {"expected": value, "observed": recipe.get(key)}
        for key, value in expected.items()
        if recipe.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{role} training recipe mismatch: {mismatches}")
    if role == "candidate":
        if recipe.get("freeze_encoder") is not False or not math.isclose(
            float(recipe.get("encoder_lr_scale", float("nan"))),
            0.1,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("candidate is not the encoder-0.1 Focus upgrade")
    elif recipe.get("freeze_encoder") is not True:
        raise ValueError("frozen comparator does not freeze its encoder")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--frozen-summary", type=Path, required=True)
    parser.add_argument("--b32-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate = json.loads(args.candidate_summary.read_text(encoding="utf-8"))
    frozen = json.loads(args.frozen_summary.read_text(encoding="utf-8"))
    b32 = json.loads(args.b32_summary.read_text(encoding="utf-8"))
    index_sha = str(candidate.get("index_sha256", ""))
    if not index_sha or b32.get("index_sha256") != index_sha:
        raise ValueError("candidate, frozen Focus, and B32 index identity mismatch")
    validate_focus(candidate, "candidate", index_sha)
    validate_focus(frozen, "frozen", index_sha)
    coverage = b32.get("coverage", {})
    if not coverage.get("every_patch_exactly_once") or int(
        coverage.get("cases", 0)
    ) < 650:
        raise ValueError("B32 summary lacks balanced exact-707 OOF coverage")

    candidate_global = candidate["segmentation"]
    frozen_global = frozen["segmentation"]
    b32_global = b32["segmentation"]
    global_diagnostic = {
        "candidate_case_dice": float(candidate_global["dice"]),
        "candidate_cohort_dice": float(candidate_global["cohort_balanced_dice"]),
        "candidate_negative_nonempty_rate": float(
            candidate_global["negative_nonempty_rate"]
        ),
        "frozen_case_dice": float(frozen_global["dice"]),
        "frozen_cohort_dice": float(frozen_global["cohort_balanced_dice"]),
        "frozen_negative_nonempty_rate": float(
            frozen_global["negative_nonempty_rate"]
        ),
        "b32_case_dice": float(b32_global["dice"]),
        "b32_cohort_dice": float(b32_global["cohort_balanced_dice"]),
    }
    finite(global_diagnostic)
    for comparator in ("frozen", "b32"):
        global_diagnostic[f"case_gain_over_{comparator}"] = (
            global_diagnostic["candidate_case_dice"]
            - global_diagnostic[f"{comparator}_case_dice"]
        )
        global_diagnostic[f"cohort_gain_over_{comparator}"] = (
            global_diagnostic["candidate_cohort_dice"]
            - global_diagnostic[f"{comparator}_cohort_dice"]
        )
    global_diagnostic["negative_nonempty_increase"] = (
        global_diagnostic["candidate_negative_nonempty_rate"]
        - global_diagnostic["frozen_negative_nonempty_rate"]
    )

    candidate_routes = candidate["source_segmentation_groups"]
    frozen_routes = frozen["source_segmentation_groups"]
    route_gate = {
        "minimum_case_gain_over_frozen": 0.002,
        "minimum_cohort_gain_over_frozen": 0.002,
        "minimum_selection_gain_over_frozen": 0.001,
        "maximum_negative_nonempty_increase": 0.02,
    }
    routes: dict[str, dict] = {}
    qualified_routes: list[str] = []
    for route in ROUTES:
        candidate_route = candidate_routes[route]
        frozen_route = frozen_routes[route]
        metrics = {
            "candidate_case_dice": float(candidate_route["dice"]),
            "candidate_cohort_dice": float(candidate_route["cohort_balanced_dice"]),
            "candidate_selection_score": float(candidate_route["selection_score"]),
            "candidate_negative_nonempty_rate": float(
                candidate_route["negative_nonempty_rate"]
            ),
            "frozen_case_dice": float(frozen_route["dice"]),
            "frozen_cohort_dice": float(frozen_route["cohort_balanced_dice"]),
            "frozen_selection_score": float(frozen_route["selection_score"]),
            "frozen_negative_nonempty_rate": float(
                frozen_route["negative_nonempty_rate"]
            ),
        }
        finite(metrics)
        metrics["case_gain_over_frozen"] = (
            metrics["candidate_case_dice"] - metrics["frozen_case_dice"]
        )
        metrics["cohort_gain_over_frozen"] = (
            metrics["candidate_cohort_dice"] - metrics["frozen_cohort_dice"]
        )
        metrics["selection_gain_over_frozen"] = (
            metrics["candidate_selection_score"] - metrics["frozen_selection_score"]
        )
        metrics["negative_nonempty_increase"] = (
            metrics["candidate_negative_nonempty_rate"]
            - metrics["frozen_negative_nonempty_rate"]
        )
        checks = {
            "case_gain": metrics["case_gain_over_frozen"]
            >= route_gate["minimum_case_gain_over_frozen"],
            "cohort_gain": metrics["cohort_gain_over_frozen"]
            >= route_gate["minimum_cohort_gain_over_frozen"],
            "selection_gain": metrics["selection_gain_over_frozen"]
            >= route_gate["minimum_selection_gain_over_frozen"],
            "negative_increase": metrics["negative_nonempty_increase"]
            <= route_gate["maximum_negative_nonempty_increase"],
        }
        qualified = all(checks.values())
        if qualified:
            qualified_routes.append(route)
        routes[route] = {
            "metrics": metrics,
            "checks": checks,
            "qualified_for_full_volume": qualified,
        }

    passed = bool(qualified_routes)
    payload = {
        "schema_version": 2,
        "created_at": datetime.now().astimezone().isoformat(),
        "selection_basis": (
            "exact case-held-out five-fold patch OOF with independently selected "
            "weak/lung epochs; route gate fixed before folds 1-4 training"
        ),
        "decision": (
            "PROMOTE_ENCODER_UPGRADE_TO_ROUTED_FULL_VOLUME_OOF"
            if passed
            else "KEEP_FROZEN_FOCUS"
        ),
        "passed": passed,
        "qualified_routes": qualified_routes,
        "mature_route": "B32_UNCHANGED",
        "candidate_summary": str(args.candidate_summary.resolve()),
        "candidate_summary_sha256": sha256(args.candidate_summary),
        "frozen_summary": str(args.frozen_summary.resolve()),
        "frozen_summary_sha256": sha256(args.frozen_summary),
        "b32_summary": str(args.b32_summary.resolve()),
        "b32_summary_sha256": sha256(args.b32_summary),
        "global_focus_diagnostic_not_a_gate": global_diagnostic,
        "route_gate": route_gate,
        "routes": routes,
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
