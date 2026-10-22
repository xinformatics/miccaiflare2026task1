#!/usr/bin/env python3
"""Build deterministic GF-Screen train/validation lists from the FLARE25 corpus.

Public challenge validation is deliberately not referenced here. Empty-label cases
are separated so they can teach the Glance gate to reject healthy volumes. The
remaining cases are split by source cohort to keep every cohort represented.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path


CHEST_PREFIXES = (
    "Chest_",
    "LymphNodes-Mediastinal-SEG_",
    "Esophagus_",
)


def cohort(case: str) -> str:
    parts = case.split("_")
    return parts[0] if parts else case


def is_empty(properties_file: Path) -> bool:
    with properties_file.open("rb") as f:
        props = pickle.load(f)
    return not any(len(v) for v in props.get("class_locations", {}).values())


def split_cases(cases: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in cases:
        groups[cohort(item["name"])].append(item)
    train, val = [], []
    rng = random.Random(seed)
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda x: x["name"])
        rng.shuffle(group)
        n_val = max(1, round(len(group) * val_fraction)) if len(group) >= 10 else 0
        val.extend(group[:n_val])
        train.extend(group[n_val:])
    return sorted(train, key=lambda x: x["name"]), sorted(val, key=lambda x: x["name"])


def write_json(path: Path, train: list[dict], val: list[dict]) -> None:
    payload = {
        "description": "FLARE25 local split; public validation excluded",
        "training": train,
        "validation": val,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=Path("../../nnUNet_raw/Dataset501_PanCancerCT"))
    p.add_argument("--properties", type=Path, default=Path("../../nnUNet_preprocessed/Dataset501_PanCancer/nnUNetPlans_3d_fullres_S4D2W32"))
    p.add_argument("--output", type=Path, default=Path("jsons/local"))
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=20260809)
    args = p.parse_args()

    raw = args.raw.resolve()
    properties = args.properties.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    images = sorted((raw / "imagesTr").glob("*_0000.nii.gz"))
    if not images:
        raise SystemExit(f"No images found under {raw / 'imagesTr'}")

    abdomen, chest, healthy = [], [], []
    for image in images:
        name = image.name.removesuffix("_0000.nii.gz")
        label = raw / "labelsTr" / f"{name}.nii.gz"
        props = properties / f"{name}.pkl"
        if not label.is_file() or not props.is_file():
            raise FileNotFoundError(f"Missing label/properties for {name}")
        item = {"image": str(image.resolve()), "label": str(label.resolve()), "name": name}
        if is_empty(props):
            healthy.append(item)
        elif name.startswith(CHEST_PREFIXES):
            chest.append(item)
        else:
            abdomen.append(item)

    for name, cases, offset in (("abdomen", abdomen, 0), ("chest", chest, 1), ("healthy", healthy, 2)):
        train, val = split_cases(cases, args.val_fraction, args.seed + offset)
        write_json(args.output / f"{name}.json", train, val)
        print(f"{name}: train={len(train)} val={len(val)} total={len(cases)}")


if __name__ == "__main__":
    main()
