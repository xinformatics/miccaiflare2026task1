#!/usr/bin/env python3
"""Validate and emit a source-routed checkpoint deployment specification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


EXPECTED_SOURCES = {
    "epoch105",
    "currentgf",
    "autopet_weak",
    "lndb_weak",
    *(f"lung_cell{i}" for i in range(8)),
}
SAFE_TOKEN = re.compile(r"^[A-Za-z0-9_.-]+$")


def _checkpoint_basename(value: object, description: str) -> str:
    filename = str(value).strip()
    if (
        not filename
        or not SAFE_TOKEN.fullmatch(filename)
        or Path(filename).name != filename
        or Path(filename).suffix != ".pt"
    ):
        raise ValueError(f"{description} must be a safe .pt basename")
    return filename


def deployment_specification(payload: dict) -> dict:
    routes = payload.get("contour_checkpoint_filenames", {})
    sources = payload.get("source_contour_routes", {})
    families = payload.get("contour_checkpoint_run_families", {})
    classifier_value = payload.get("classification_checkpoint_filename", "")
    if payload.get("selected_strategy") != "source_routed" or not classifier_value:
        raise ValueError("selection is not a source-routed strategy")
    classifier = _checkpoint_basename(classifier_value, "classification checkpoint")
    if not isinstance(routes, dict) or not routes:
        raise ValueError("source-routed selection has no contour routes")
    if set(sources) != EXPECTED_SOURCES:
        raise ValueError("source-routed selection has incomplete source inventory")
    if set(sources.values()) != set(routes):
        raise ValueError("source routes do not use every and only declared route")
    if set(families) != set(routes):
        raise ValueError("route checkpoint-family coverage mismatch")
    if set(families.values()) - {"primary", "discovery"}:
        raise ValueError("invalid route checkpoint family")
    normalized_routes = {}
    for name, filename in routes.items():
        if not isinstance(name, str) or not SAFE_TOKEN.fullmatch(name):
            raise ValueError("route name must be a safe token")
        normalized_routes[name] = _checkpoint_basename(filename, "route checkpoint")
    return {
        "classification_checkpoint_filename": classifier,
        "routes": {
            name: {"filename": normalized_routes[name], "family": str(families[name])}
            for name in sorted(routes)
        },
        "sources": {name: str(sources[name]) for name in sorted(sources)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    result = deployment_specification(
        json.loads(args.selection.read_text(encoding="utf-8"))
    )
    print("CLASS\t" + result["classification_checkpoint_filename"])
    for name, route in result["routes"].items():
        print(f"ROUTE\t{name}\t{route['filename']}\t{route['family']}")
    for name, route in result["sources"].items():
        print(f"SOURCE\t{name}\t{route}")


if __name__ == "__main__":
    main()
