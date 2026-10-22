#!/usr/bin/env python3
"""Create a fixed, stratified held-out healthy proxy cohort.

Selection depends only on filenames, cohort prefixes, and a predeclared seed;
no model predictions or labels are consulted.  Symlinks avoid duplicating CTs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def cohort(path: Path) -> str:
    return path.name.split("_", 1)[0]


def proportional_quotas(groups: dict[str, list[Path]], total: int) -> dict[str, int]:
    available = sum(len(paths) for paths in groups.values())
    if total < 1 or total > available:
        raise ValueError(f"total must be in [1, {available}], got {total}")
    exact = {name: total * len(paths) / available for name, paths in groups.items()}
    quotas = {name: int(value) for name, value in exact.items()}
    remaining = total - sum(quotas.values())
    order = sorted(groups, key=lambda name: (-(exact[name] - quotas[name]), name))
    for name in order[:remaining]:
        quotas[name] += 1
    return quotas


def select_paths(paths: list[Path], total: int, seed: str) -> tuple[list[Path], dict[str, int]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        groups[cohort(path)].append(path)
    quotas = proportional_quotas(groups, total)
    selected: list[Path] = []
    for name, candidates in sorted(groups.items()):
        ranked = sorted(
            candidates,
            key=lambda path: hashlib.sha256(
                f"{seed}\0{name}\0{path.name}".encode("utf-8")
            ).hexdigest(),
        )
        selected.extend(ranked[: quotas[name]])
    return sorted(selected, key=lambda path: path.name), quotas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--total", type=int, default=32)
    parser.add_argument("--seed", default="miccai26-biomedparse-healthy-v1")
    args = parser.parse_args()

    paths = sorted(args.images.glob("*.nii.gz"))
    if not paths:
        raise FileNotFoundError(f"no NIfTI images in {args.images}")
    selected, quotas = select_paths(paths, args.total, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source in selected:
        destination = args.output_dir / source.name
        if destination.is_symlink():
            if destination.resolve() != source.resolve():
                raise RuntimeError(f"wrong existing link: {destination}")
        elif destination.exists():
            raise FileExistsError(destination)
        else:
            destination.symlink_to(source.resolve())
    unexpected = sorted(
        path.name for path in args.output_dir.glob("*.nii.gz")
        if path.name not in {source.name for source in selected}
    )
    if unexpected:
        raise RuntimeError(f"unexpected existing proxy images: {unexpected[:5]}")

    manifest = {
        "source": str(args.images.resolve()),
        "selection_is_prediction_blind": True,
        "seed": args.seed,
        "total": len(selected),
        "available": len(paths),
        "quotas": quotas,
        "cases": [source.name for source in selected],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
