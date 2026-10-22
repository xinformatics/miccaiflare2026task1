#!/usr/bin/env python3
"""Validate an inference cache marker against exact checkpoint provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def named_paths(values: list[str]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = defaultdict(list)
    for value in values:
        if "=" not in value:
            raise ValueError(f"route must use NAME=PATH: {value}")
        name, raw_path = value.split("=", 1)
        name, raw_path = name.strip(), raw_path.strip()
        if not name or not raw_path:
            raise ValueError(f"invalid route: {value}")
        result[name].append(Path(raw_path))
    return dict(result)


def source_routes(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"source route must use SOURCE=NAME: {value}")
        source, route = (part.strip() for part in value.split("=", 1))
        if not source or not route or source in result:
            raise ValueError(f"invalid or duplicate source route: {value}")
        result[source] = route
    return result


def path_identity(paths: list[Path]) -> tuple[list[str], list[str]]:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint files are missing: {missing}")
    return (
        [str(path.resolve()) for path in paths],
        [sha256(path) for path in paths],
    )


def validate(
    payload: dict,
    checkpoints: list[Path],
    contour_checkpoints: list[Path] | None = None,
    contour_input_mode: str = "",
    candidate_contour_checkpoints: list[Path] | None = None,
    routed_checkpoints: dict[str, list[Path]] | None = None,
    focus_routed_checkpoints: dict[str, list[Path]] | None = None,
    focus_routed_input_mode: str = "",
    source_to_route: dict[str, str] | None = None,
    expected_candidates: int = 0,
    classifier_fusion: Path | None = None,
    routed_classifier_checkpoints: dict[str, list[Path]] | None = None,
    source_to_classifier_route: dict[str, str] | None = None,
) -> dict[str, bool]:
    contour_checkpoints = contour_checkpoints or []
    candidate_contour_checkpoints = candidate_contour_checkpoints or []
    routed_checkpoints = routed_checkpoints or {}
    focus_routed_checkpoints = focus_routed_checkpoints or {}
    source_to_route = source_to_route or {}
    routed_classifier_checkpoints = routed_classifier_checkpoints or {}
    source_to_classifier_route = source_to_classifier_route or {}
    candidate_route_names = set(routed_checkpoints)
    focus_route_names = set(focus_routed_checkpoints)
    overlap = sorted(candidate_route_names & focus_route_names)
    if overlap:
        raise ValueError(
            "candidate-refiner and Focus contour route names overlap: "
            f"{overlap}"
        )
    selected_routes = {**routed_checkpoints, **focus_routed_checkpoints}
    override_count = sum(
        bool(value)
        for value in (
            contour_checkpoints,
            candidate_contour_checkpoints,
            selected_routes,
        )
    )
    if override_count > 1:
        raise ValueError("contour override families are mutually exclusive")
    if bool(selected_routes) != bool(source_to_route):
        raise ValueError("routed checkpoints and source map are required together")
    if selected_routes and set(selected_routes) != set(source_to_route.values()):
        raise ValueError("routed checkpoint names do not match the source map")
    if bool(routed_classifier_checkpoints) != bool(source_to_classifier_route):
        raise ValueError(
            "routed classifier checkpoints and source map are required together"
        )
    if routed_classifier_checkpoints and set(routed_classifier_checkpoints) != set(
        source_to_classifier_route.values()
    ):
        raise ValueError(
            "routed classifier checkpoint names do not match the source map"
        )
    if classifier_fusion is not None and (
        not contour_checkpoints
        or candidate_contour_checkpoints
        or selected_routes
        or routed_classifier_checkpoints
    ):
        raise ValueError(
            "classifier fusion requires one global Focus contour family"
        )

    checkpoint_paths, checkpoint_hashes = path_identity(checkpoints)
    contour_paths, contour_hashes = path_identity(contour_checkpoints)
    candidate_paths, candidate_hashes = path_identity(candidate_contour_checkpoints)
    recorded_routes = payload.get("routed_candidate_contours", {})
    recorded_focus_routes = payload.get("routed_focus_contours", {})
    recorded_classifier_routes = payload.get("routed_classifiers", {})
    recorded_fusion = payload.get("classifier_fusion", {})
    if classifier_fusion is None:
        fusion_identity = recorded_fusion in ({}, None)
    else:
        fusion_paths, fusion_hashes = path_identity([classifier_fusion])
        fusion_identity = bool(
            isinstance(recorded_fusion, dict)
            and recorded_fusion.get("decision") == fusion_paths[0]
            and recorded_fusion.get("decision_sha256") == fusion_hashes[0]
            and recorded_fusion.get("focus_input_mode") == contour_input_mode
            and isinstance(recorded_fusion.get("feature_names"), list)
            and len(recorded_fusion["feature_names"]) > 0
        )
    route_checks: dict[str, bool] = {}
    for name, paths in routed_checkpoints.items():
        expected_paths, expected_hashes = path_identity(paths)
        record = recorded_routes.get(name, {}) if isinstance(recorded_routes, dict) else {}
        groups = record.get("source_condition_groups", [])
        route_checks[name] = bool(
            record.get("checkpoints") == expected_paths
            and record.get("checkpoint_sha256") == expected_hashes
            and isinstance(groups, list)
            and len(groups) == len(paths)
            and all(isinstance(group, list) for group in groups)
        )
    focus_route_checks: dict[str, bool] = {}
    for name, paths in focus_routed_checkpoints.items():
        expected_paths, expected_hashes = path_identity(paths)
        record = (
            recorded_focus_routes.get(name, {})
            if isinstance(recorded_focus_routes, dict)
            else {}
        )
        focus_route_checks[name] = bool(
            record.get("checkpoints") == expected_paths
            and record.get("checkpoint_sha256") == expected_hashes
            and record.get("input_modes")
            == [focus_routed_input_mode] * len(paths)
        )
    classifier_route_checks: dict[str, bool] = {}
    for name, paths in routed_classifier_checkpoints.items():
        expected_paths, expected_hashes = path_identity(paths)
        record = (
            recorded_classifier_routes.get(name, {})
            if isinstance(recorded_classifier_routes, dict)
            else {}
        )
        groups = record.get("source_condition_groups", [])
        classifier_route_checks[name] = bool(
            record.get("checkpoints") == expected_paths
            and record.get("checkpoint_sha256") == expected_hashes
            and isinstance(groups, list)
            and len(groups) == len(paths)
            and all(isinstance(group, list) for group in groups)
        )
    checks = {
        "candidate_count": expected_candidates <= 0
        or int(payload.get("candidates", -1)) == expected_candidates,
        "classifier_paths": payload.get("checkpoints", []) == checkpoint_paths,
        "classifier_hashes": payload.get("checkpoint_sha256", [])
        == checkpoint_hashes,
        "classifier_layouts": "source_condition_groups" not in payload
        or (
            isinstance(payload.get("source_condition_groups"), list)
            and len(payload.get("source_condition_groups", [])) == len(checkpoints)
            and all(
                isinstance(group, list)
                for group in payload.get("source_condition_groups", [])
            )
        ),
        "focus_contour_paths": payload.get("contour_checkpoints", [])
        == contour_paths,
        "focus_contour_hashes": payload.get("contour_checkpoint_sha256", [])
        == contour_hashes,
        "focus_contour_modes": payload.get("contour_input_modes", [])
        == ([contour_input_mode] * len(contour_checkpoints)),
        "candidate_contour_paths": payload.get(
            "candidate_contour_checkpoints", []
        )
        == candidate_paths,
        "candidate_contour_hashes": payload.get(
            "candidate_contour_checkpoint_sha256", []
        )
        == candidate_hashes,
        "routed_inventory": isinstance(recorded_routes, dict)
        and set(recorded_routes) == set(routed_checkpoints),
        "routed_checkpoints": all(route_checks.values()),
        "routed_focus_inventory": isinstance(recorded_focus_routes, dict)
        and set(recorded_focus_routes) == set(focus_routed_checkpoints),
        "routed_focus_checkpoints": all(focus_route_checks.values()),
        "source_routes": payload.get("source_contour_routes", {})
        == dict(sorted(source_to_route.items())),
        "routed_classifier_inventory": isinstance(recorded_classifier_routes, dict)
        and set(recorded_classifier_routes) == set(routed_classifier_checkpoints),
        "routed_classifier_checkpoints": all(classifier_route_checks.values()),
        "source_classifier_routes": payload.get("source_classifier_routes", {})
        == dict(sorted(source_to_classifier_route.items())),
        "classifier_fusion": fusion_identity,
    }
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--contour-checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--contour-input-mode", default="")
    parser.add_argument(
        "--candidate-contour-checkpoint", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--candidate-contour-route", action="append", default=[], metavar="NAME=PATH"
    )
    parser.add_argument(
        "--focus-contour-route", action="append", default=[], metavar="NAME=PATH"
    )
    parser.add_argument("--focus-routed-input-mode", default="")
    parser.add_argument(
        "--source-contour-route", action="append", default=[], metavar="SOURCE=NAME"
    )
    parser.add_argument("--classifier-fusion", type=Path)
    parser.add_argument(
        "--classifier-route", action="append", default=[], metavar="NAME=PATH"
    )
    parser.add_argument(
        "--source-classifier-route",
        action="append",
        default=[],
        metavar="SOURCE=NAME",
    )
    parser.add_argument("--expected-candidates", type=int, default=0)
    args = parser.parse_args()
    if args.expected_candidates < 0:
        parser.error("expected candidates must be nonnegative")
    if not args.marker.is_file():
        raise FileNotFoundError(args.marker)
    payload = json.loads(args.marker.read_text(encoding="utf-8"))
    checks = validate(
        payload,
        args.checkpoint,
        args.contour_checkpoint,
        args.contour_input_mode,
        args.candidate_contour_checkpoint,
        named_paths(args.candidate_contour_route),
        named_paths(args.focus_contour_route),
        args.focus_routed_input_mode,
        source_routes(args.source_contour_route),
        args.expected_candidates,
        args.classifier_fusion,
        named_paths(args.classifier_route),
        source_routes(args.source_classifier_route),
    )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit(f"inference marker provenance failed: {failed}")
    print(json.dumps({"marker": str(args.marker.resolve()), "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
