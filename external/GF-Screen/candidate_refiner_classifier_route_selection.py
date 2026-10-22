#!/usr/bin/env python3
"""Emit validated source-classifier deployment routes from exact707 OOF."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


GROUP_SOURCES = {
    "mature": ("epoch105", "currentgf"),
    "weak_sources": ("autopet_weak", "lndb_weak"),
    "lung_cells": tuple(f"lung_cell{index}" for index in range(8)),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def route_records(payload: dict, fold: int | None = None) -> list[tuple]:
    if payload.get("decision") != "QUALIFY_SOURCE_ROUTED_CLASSIFIER":
        raise ValueError("source-routed classifier did not qualify")
    index = Path(str(payload.get("index", "")))
    predictions = Path(str(payload.get("predictions", "")))
    if (
        payload.get("selection_basis")
        != "source-family checkpoint routing on exact707 grouped held-out patches"
        or payload.get("public_read") is not False
        or payload.get("hidden_read") is not False
        or payload.get("submission_created") is not False
        or not index.is_file()
        or sha256(index) != payload.get("index_sha256")
        or not predictions.is_file()
        or sha256(predictions) != payload.get("predictions_sha256")
        or set(payload.get("checks", {}))
        != {"average_precision_gain", "f0p5_gain", "source_nonregression"}
        or not all(value is True for value in payload.get("checks", {}).values())
    ):
        raise ValueError("source-routed classifier provenance or gates are incomplete")
    selections = payload.get("selections", [])
    if sorted(int(item.get("fold", -1)) for item in selections) != list(range(5)):
        raise ValueError("classifier route decision lacks exact five-fold selections")
    records = []
    for item in sorted(selections, key=lambda value: int(value["fold"])):
        item_fold = int(item["fold"])
        if fold is not None and item_fold != fold:
            continue
        routes = item.get("routes", {})
        if set(routes) != set(GROUP_SOURCES):
            raise ValueError(f"classifier route group mismatch in fold {item_fold}")
        for group in sorted(GROUP_SOURCES):
            route = routes[group]
            checkpoint = Path(route["checkpoint"])
            if not checkpoint.is_file() or sha256(checkpoint) != route.get("checkpoint_sha256"):
                raise ValueError(f"classifier route checkpoint identity mismatch: {checkpoint}")
            records.append(("ROUTE", item_fold, group, str(checkpoint.resolve())))
    for group, sources in sorted(GROUP_SOURCES.items()):
        for source in sources:
            records.append(("SOURCE", "-", source, group))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(5))
    args = parser.parse_args()
    payload = json.loads(args.decision.read_text(encoding="utf-8"))
    for record in route_records(payload, args.fold):
        print("\t".join(map(str, record)))


if __name__ == "__main__":
    main()
