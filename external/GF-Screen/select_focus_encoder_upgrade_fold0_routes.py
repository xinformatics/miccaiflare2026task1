#!/usr/bin/env python3
"""Gate a fold-0 encoder upgrade on the Focus routes used at deployment."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path

import torch

from summarize_focus_epoch_routes import sha256


DEPLOYED_ROUTES = ("weak_sources", "lung_cells")


def finite(values: dict[str, float]) -> None:
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("non-finite fold-0 route metric")


def frozen_fold_metrics(summary: dict, route: str) -> dict[str, float]:
    records = [record for record in summary["fold_records"] if int(record["fold"]) == 0]
    if len(records) != 1:
        raise ValueError("frozen route summary lacks one fold-0 record")
    record = records[0]
    selected = record["selected"][route]
    epoch = int(selected["epoch"])
    trajectories = [row for row in record["trajectory"] if int(row["epoch"]) == epoch]
    if len(trajectories) != 1:
        raise ValueError(f"frozen {route} selected epoch lacks trajectory metrics")
    metrics = trajectories[0]["groups"][route]
    for key in ("selection_score", "case_balanced_dice", "negative_nonempty_rate"):
        if not math.isclose(
            float(selected[key]), float(metrics[key]), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"frozen {route} selected/trajectory metric mismatch")
    return {
        "epoch": epoch,
        "selection_score": float(metrics["selection_score"]),
        "case_balanced_dice": float(metrics["case_balanced_dice"]),
        "cohort_balanced_dice": float(metrics["cohort_balanced_dice"]),
        "negative_nonempty_rate": float(metrics["negative_nonempty_rate"]),
    }


def validate_candidate_selection(selection: dict, plan: dict) -> None:
    training = plan["training"]
    if (
        int(selection.get("fold", -1)) != 0
        or int(selection.get("expected_epochs", -1)) != int(training["epochs"]) + 1
        or int(selection.get("seed", -1)) != int(training["seed"])
        or selection.get("index_sha256") != plan["index_sha256"]
        or set(selection.get("selected", {}))
        != {"mature", "weak_sources", "lung_cells"}
        or selection.get("public_read")
        or selection.get("hidden_read")
        or selection.get("submission_created")
    ):
        raise ValueError("candidate route selection identity mismatch")

    expected_config = dict(training)
    for route, selected in selection["selected"].items():
        checkpoint_path = Path(selected["checkpoint"])
        prediction_path = Path(selected["prediction"])
        audit_path = Path(selected["audit"])
        if sha256(checkpoint_path) != selected["checkpoint_sha256"]:
            raise ValueError(f"candidate {route} checkpoint hash mismatch")
        if sha256(prediction_path) != selected["prediction_sha256"]:
            raise ValueError(f"candidate {route} prediction hash mismatch")
        if sha256(audit_path) != selected["audit_sha256"]:
            raise ValueError(f"candidate {route} audit hash mismatch")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (
            int(audit.get("epoch", -1)) != int(selected["epoch"])
            or audit.get("checkpoint_sha256") != selected["checkpoint_sha256"]
            or audit.get("prediction_sha256") != selected["prediction_sha256"]
            or audit.get("groups", {}).get(route) != selected["metrics"]
        ):
            raise ValueError(f"candidate {route} audit identity mismatch")
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False, mmap=True
        )
        config = checkpoint.get("config", {})
        mismatches = {
            key: {"expected": value, "observed": config.get(key)}
            for key, value in expected_config.items()
            if config.get(key) != value
        }
        if (
            mismatches
            or int(checkpoint.get("fold", -1)) != 0
            or int(checkpoint.get("folds", -1)) != 5
            or int(checkpoint.get("epoch", -1)) != int(selected["epoch"])
            or checkpoint.get("index_sha256") != plan["index_sha256"]
        ):
            raise ValueError(
                f"candidate {route} checkpoint recipe/identity mismatch: {mismatches}"
            )


def select(plan: dict, candidate: dict, frozen: dict) -> dict:
    validate_candidate_selection(candidate, plan)
    if (
        frozen.get("index_sha256") != plan["index_sha256"]
        or int(frozen.get("folds", -1)) != 5
        or int(frozen.get("seed", -1)) != int(plan["training"]["seed"])
        or int(frozen.get("expected_epochs", -1))
        != int(plan["training"]["epochs"]) + 1
        or frozen.get("training_recipe", {}).get("freeze_encoder") is not True
        or frozen.get("training_recipe", {}).get("input_mode")
        != plan["training"]["input_mode"]
        or frozen.get("public_read")
        or frozen.get("hidden_read")
        or frozen.get("submission_created")
    ):
        raise ValueError("frozen Focus route summary identity mismatch")

    gate = plan["route_gate"]
    routes: dict[str, dict] = {}
    qualified_routes: list[str] = []
    for route in DEPLOYED_ROUTES:
        candidate_entry = candidate["selected"][route]
        candidate_metrics = candidate_entry["metrics"]
        baseline = frozen_fold_metrics(frozen, route)
        metrics = {
            "candidate_epoch": int(candidate_entry["epoch"]),
            "candidate_selection_score": float(candidate_metrics["selection_score"]),
            "candidate_case_dice": float(candidate_metrics["case_balanced_dice"]),
            "candidate_cohort_dice": float(candidate_metrics["cohort_balanced_dice"]),
            "candidate_negative_nonempty_rate": float(
                candidate_metrics["negative_nonempty_rate"]
            ),
            "frozen_epoch": int(baseline["epoch"]),
            "frozen_selection_score": float(baseline["selection_score"]),
            "frozen_case_dice": float(baseline["case_balanced_dice"]),
            "frozen_cohort_dice": float(baseline["cohort_balanced_dice"]),
            "frozen_negative_nonempty_rate": float(
                baseline["negative_nonempty_rate"]
            ),
        }
        finite({key: float(value) for key, value in metrics.items()})
        metrics["selection_gain"] = (
            metrics["candidate_selection_score"] - metrics["frozen_selection_score"]
        )
        metrics["case_gain"] = metrics["candidate_case_dice"] - metrics["frozen_case_dice"]
        metrics["cohort_gain"] = (
            metrics["candidate_cohort_dice"] - metrics["frozen_cohort_dice"]
        )
        metrics["negative_nonempty_increase"] = (
            metrics["candidate_negative_nonempty_rate"]
            - metrics["frozen_negative_nonempty_rate"]
        )
        checks = {
            "case_gain": metrics["case_gain"] >= float(gate["minimum_case_gain"]),
            "cohort_gain": metrics["cohort_gain"]
            >= float(gate["minimum_cohort_gain"]),
            "selection_gain": metrics["selection_gain"]
            >= float(gate["minimum_selection_gain"]),
            "negative_increase": metrics["negative_nonempty_increase"]
            <= float(gate["maximum_negative_nonempty_increase"]),
        }
        qualified = all(checks.values())
        if qualified:
            qualified_routes.append(route)
        routes[route] = {
            "metrics": metrics,
            "checks": checks,
            "qualified_for_fivefold": qualified,
        }

    passed = bool(qualified_routes)
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "selection_basis": plan["selection_basis"],
        "decision": "PROMOTE_ROUTES_TO_FIVEFOLD" if passed else "KEEP_FROZEN_FOCUS",
        "passed": passed,
        "qualified_routes": qualified_routes,
        "mature_route": "B32_UNCHANGED",
        "route_gate": gate,
        "routes": routes,
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--candidate-routes", type=Path, required=True)
    parser.add_argument("--frozen-routes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate_routes.read_text(encoding="utf-8"))
    frozen = json.loads(args.frozen_routes.read_text(encoding="utf-8"))
    composite_path = Path(plan["qualified_composite_decision"])
    if (
        plan.get("public_read")
        or plan.get("hidden_read")
        or plan.get("submission_created")
        or Path(plan["frozen_routes"]).resolve() != args.frozen_routes.resolve()
        or sha256(args.frozen_routes) != plan["frozen_routes_sha256"]
        or sha256(composite_path) != plan["qualified_composite_decision_sha256"]
    ):
        raise ValueError("route plan provenance changed after the gate was frozen")
    composite = json.loads(composite_path.read_text(encoding="utf-8"))
    if composite.get("decision") != "QUALIFY_COMPOSITE_FOCUS_B32_TTA_FULL_VOLUME":
        raise ValueError("qualified composite route is unavailable")
    payload = select(plan, candidate, frozen)
    payload.update(
        {
            "plan": str(args.plan.resolve()),
            "plan_sha256": sha256(args.plan),
            "candidate_routes": str(args.candidate_routes.resolve()),
            "candidate_routes_sha256": sha256(args.candidate_routes),
            "frozen_routes": str(args.frozen_routes.resolve()),
            "frozen_routes_sha256": sha256(args.frozen_routes),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
