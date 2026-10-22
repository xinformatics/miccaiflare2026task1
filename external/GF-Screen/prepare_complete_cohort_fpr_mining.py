#!/usr/bin/env python3
"""Create a deterministic, label-safe proposal-mining cohort.

Only source datasets explicitly listed in the documentation-backed complete
label manifest are eligible.  The staged directory contains image symlinks for
GF inference; labels remain outside that directory and are recorded only in
the manifest used to assign proposal-component targets afterward.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from inference_naming import prediction_name


def source_dataset_prefix(name: str) -> str:
    tokens = str(name or "unknown").split("_")
    first = tokens[0] or "unknown"
    if len(tokens) > 1 and "-" not in first:
        second = tokens[1]
        if second and not second.isdigit():
            return f"{first}_{second}"
    return first


def identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def assert_training_path(path: Path) -> None:
    parts = {part.casefold() for part in path.resolve().parts}
    forbidden = {
        "validation",
        "validation-public-images",
        "validation-public-labels",
        "validation-hidden-images",
    }
    if parts & forbidden:
        raise ValueError(f"validation path is forbidden in training mining: {path}")


def load_items(path: Path, region: str) -> list[dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("training")
    if not isinstance(items, list):
        raise ValueError(f"{path}: missing training list")
    output = []
    for item in items:
        if not all(item.get(key) for key in ("name", "image", "label")):
            raise ValueError(f"{path}: incomplete training item")
        image = Path(item["image"])
        label = Path(item["label"])
        assert_training_path(image)
        assert_training_path(label)
        output.append(
            {
                "name": str(item["name"]),
                "image": str(image.resolve()),
                "label": str(label.resolve()),
                "cohort": f"{region}:{source_dataset_prefix(item['name'])}",
            }
        )
    return output


def deterministic_rank(seed: int, name: str) -> str:
    return hashlib.sha256(f"{seed}:{name}".encode("utf-8")).hexdigest()


def select_cases(
    items: list[dict[str, str]], complete: set[str], per_cohort: int, seed: int
) -> list[dict[str, str]]:
    groups: dict[str, list[dict[str, str]]] = {key: [] for key in sorted(complete)}
    for item in items:
        if item["cohort"] in groups:
            groups[item["cohort"]].append(item)
    missing = [cohort for cohort, values in groups.items() if len(values) < per_cohort]
    if missing:
        raise ValueError(f"complete cohorts have fewer than {per_cohort} cases: {missing}")
    selected = []
    for cohort, values in groups.items():
        ordered = sorted(values, key=lambda item: deterministic_rank(seed, item["name"]))
        selected.extend(ordered[:per_cohort])
    return sorted(selected, key=lambda item: (item["cohort"], item["name"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--abdomen-json", type=Path, required=True)
    parser.add_argument("--chest-json", type=Path, required=True)
    parser.add_argument("--complete-manifest", type=Path, required=True)
    parser.add_argument("--output-images", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--per-cohort", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    if args.per_cohort < 1:
        parser.error("per-cohort must be positive")

    complete_payload = json.loads(args.complete_manifest.read_text(encoding="utf-8"))
    complete_values = complete_payload.get("complete_label_cohorts")
    if not isinstance(complete_values, list) or not complete_values:
        raise ValueError("complete manifest contains no cohorts")
    complete = set(map(str, complete_values))
    items = load_items(args.abdomen_json, "abdomen") + load_items(
        args.chest_json, "chest"
    )
    selected = select_cases(items, complete, args.per_cohort, args.seed)

    args.output_images.mkdir(parents=True, exist_ok=True)
    basenames: set[str] = set()
    records = []
    for item in selected:
        image = Path(item["image"])
        label = Path(item["label"])
        if not image.is_file() or not label.is_file():
            raise FileNotFoundError(image if not image.is_file() else label)
        if image.name in basenames:
            raise ValueError(f"duplicate staged basename: {image.name}")
        basenames.add(image.name)
        staged = args.output_images / image.name
        if staged.is_symlink() or staged.exists():
            if staged.resolve() != image.resolve():
                raise FileExistsError(f"incorrect existing stage target: {staged}")
        else:
            staged.symlink_to(image.resolve())
        records.append(
            {
                **item,
                "staged_image": str(staged.absolute()),
                "prediction_name": prediction_name(image.name),
                "complete_label": True,
            }
        )

    expected = len(complete) * args.per_cohort
    if len(records) != expected:
        raise RuntimeError(f"expected {expected} staged cases, found {len(records)}")
    payload = {
        "description": (
            "Deterministic GF raw-proposal mining set from documentation-verified "
            "complete-label training cohorts; public/hidden validation excluded."
        ),
        "seed": args.seed,
        "per_cohort": args.per_cohort,
        "cases": len(records),
        "cohorts": sorted(complete),
        "source_files": {
            "abdomen": identity(args.abdomen_json),
            "chest": identity(args.chest_json),
            "complete_manifest": identity(args.complete_manifest),
        },
        "items": records,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "cases": len(records),
                "cohorts": len(complete),
                "output_images": str(args.output_images.resolve()),
                "output_manifest": str(args.output_manifest.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
