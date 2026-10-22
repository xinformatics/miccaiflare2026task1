#!/usr/bin/env python3
"""Stage the disjoint Task-2 LCT cohort for GF-Screen fine-tuning.

The Task-2 NPZ images are already region-windowed to [0, 255].  This script
maps that interval onto the corresponding GF input window so GF's existing
ScaleIntensityRanged transform recovers the original normalized contrast.  It
preserves native geometry, binarizes the instance labels, groups train/holdout
by LCT patient token, and extends the immutable Task-1 manifests without
duplicating their 8,275 overlapping cases.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk


GF_WINDOWS = {
    "abdomen": (-175.0, 250.0),
    "chest": (-900.0, 650.0),
}


@dataclass(frozen=True)
class Case:
    source: str
    name: str
    patient: str
    source_region: str
    gf_region: str
    split: str
    image: str
    label: str


def atomic_json(payload: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def patient_split(patient: str, seed: int, holdout_fraction: float) -> str:
    digest = hashlib.sha256(f"{seed}:{patient}".encode()).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return "validation" if value < holdout_fraction else "training"


def source_region(path: Path) -> str:
    marker = "__"
    if marker not in path.stem:
        raise ValueError(f"LCT filename lacks region suffix: {path.name}")
    return path.stem.rsplit(marker, 1)[1]


def staged_name(path: Path) -> str:
    if not path.stem.startswith("LCT_"):
        raise ValueError(f"unexpected LCT case name: {path.name}")
    # This yields one stable cohort key, WholeBody_LCT, in GF's source parser.
    return "WholeBody_LCT_" + path.stem.removeprefix("LCT_")


def patient_token(path: Path) -> str:
    tokens = path.stem.split("_")
    if len(tokens) < 3 or tokens[0] != "LCT":
        raise ValueError(f"cannot recover patient token: {path.name}")
    return tokens[1]


def make_case(
    source: Path,
    output_root: Path,
    seed: int,
    holdout_fraction: float,
) -> Case | None:
    src_region = source_region(source)
    if src_region == "brain":
        return None
    gf_region = "chest" if src_region == "lung" else "abdomen"
    name = staged_name(source)
    patient = patient_token(source)
    return Case(
        source=str(source.resolve()),
        name=name,
        patient=patient,
        source_region=src_region,
        gf_region=gf_region,
        split=patient_split(patient, seed, holdout_fraction),
        image=str((output_root / "imagesTr" / f"{name}_0000.nii.gz").resolve()),
        label=str((output_root / "labelsTr" / f"{name}.nii.gz").resolve()),
    )


def write_image_atomic(image: sitk.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.nii.gz")
    sitk.WriteImage(image, str(temporary), True)
    os.replace(temporary, destination)


def valid_existing(case: Case) -> bool:
    image_path = Path(case.image)
    label_path = Path(case.label)
    if not image_path.is_file() or not label_path.is_file():
        return False
    image = sitk.ReadImage(str(image_path))
    label = sitk.ReadImage(str(label_path))
    if image.GetSize() != label.GetSize():
        return False
    values = np.unique(sitk.GetArrayViewFromImage(label))
    return len(values) <= 2 and set(map(int, values)).issubset({0, 1})


def convert(case: Case, overwrite: bool) -> dict[str, object]:
    if not overwrite and valid_existing(case):
        return {"name": case.name, "status": "existing"}

    with np.load(case.source, allow_pickle=False) as payload:
        required = {"imgs", "gts", "spacing", "direction", "origin"}
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"{case.source}: missing keys {sorted(missing)}")
        normalized = np.asarray(payload["imgs"], dtype=np.float32) / 255.0
        labels = np.asarray(payload["gts"]) > 0
        spacing = tuple(map(float, payload["spacing"]))
        direction = tuple(map(float, payload["direction"]))
        origin = tuple(map(float, payload["origin"]))

    if normalized.ndim != 3 or normalized.shape != labels.shape:
        raise ValueError(f"{case.source}: incompatible image/label shape")
    if not np.isfinite(normalized).all() or not labels.any():
        raise ValueError(f"{case.source}: invalid image or empty label")
    low, high = GF_WINDOWS[case.gf_region]
    hu_proxy = normalized * (high - low) + low

    image = sitk.GetImageFromArray(hu_proxy.astype(np.float32, copy=False))
    label = sitk.GetImageFromArray(labels.astype(np.uint8, copy=False))
    for output in (image, label):
        output.SetSpacing(spacing)
        output.SetDirection(direction)
        output.SetOrigin(origin)
    write_image_atomic(image, Path(case.image))
    write_image_atomic(label, Path(case.label))
    return {
        "name": case.name,
        "status": "written",
        "shape_zyx": list(normalized.shape),
        "foreground_voxels": int(labels.sum()),
    }


def extend_manifest(base: Path, cases: list[Case], region: str) -> dict[str, object]:
    payload = json.loads(base.read_text(encoding="utf-8"))
    original_names = {
        item.get("name")
        for split in ("training", "validation")
        for item in payload.get(split, [])
    }
    selected = [case for case in cases if case.gf_region == region]
    overlap = sorted(case.name for case in selected if case.name in original_names)
    if overlap:
        raise ValueError(f"staged names overlap base manifest: {overlap[:5]}")
    for split in ("training", "validation"):
        additions = [
            {"image": case.image, "label": case.label, "name": case.name}
            for case in selected
            if case.split == split
        ]
        payload.setdefault(split, []).extend(additions)
    payload["description"] = (
        str(payload.get("description", ""))
        + f" Extended with patient-grouped Task-2 LCT {region} cases."
    ).strip()
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-abdomen-json", type=Path, required=True)
    parser.add_argument("--base-chest-json", type=Path, required=True)
    parser.add_argument("--output-abdomen-json", type=Path, required=True)
    parser.add_argument("--output-chest-json", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--holdout-fraction", type=float, default=0.10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.holdout_fraction < 0.5:
        raise SystemExit("--holdout-fraction must lie in (0, 0.5)")
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    sources = sorted(args.source.glob("*.npz"))
    if len(sources) != 950 and not args.limit:
        raise SystemExit(f"expected exactly 950 LCT NPZs, found {len(sources)}")
    if args.limit:
        sources = sources[: args.limit]
    cases = [
        case
        for source in sources
        if (case := make_case(source, args.output_root, args.seed, args.holdout_fraction))
        is not None
    ]

    results: list[dict[str, object]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(convert, case, args.overwrite) for case in cases]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            results.append(future.result())
            if index % 25 == 0 or index == len(futures):
                print(f"converted {index}/{len(futures)}", flush=True)

    if not args.limit:
        atomic_json(
            extend_manifest(args.base_abdomen_json, cases, "abdomen"),
            args.output_abdomen_json,
        )
        atomic_json(
            extend_manifest(args.base_chest_json, cases, "chest"),
            args.output_chest_json,
        )
    counts: dict[str, int] = {}
    for case in cases:
        key = f"{case.gf_region}_{case.split}"
        counts[key] = counts.get(key, 0) + 1
    inventory_digest = hashlib.sha256(
        "\n".join(f"{case.source}|{case.name}|{case.split}" for case in cases).encode()
    ).hexdigest()
    audit = {
        "schema_version": 1,
        "status": "COMPLETE",
        "source": str(args.source.resolve()),
        "source_npz_count": len(sources),
        "staged_case_count": len(cases),
        "excluded_brain_cases": len(sources) - len(cases),
        "unique_patients": len({case.patient for case in cases}),
        "split_seed": args.seed,
        "holdout_fraction": args.holdout_fraction,
        "counts": counts,
        "inventory_sha256": inventory_digest,
        "window_contract": {
            "source": "region-windowed [0,255] Task-2 NPZ",
            "target": GF_WINDOWS,
            "purpose": "recover identical normalized contrast under GF transforms",
        },
        "label_contract": "instance ids binarized; foreground positive; background not declared complete",
        "cases": [asdict(case) for case in cases],
        "conversion": {
            "written": sum(result["status"] == "written" for result in results),
            "existing": sum(result["status"] == "existing" for result in results),
        },
    }
    atomic_json(audit, args.audit)
    print(json.dumps({key: audit[key] for key in ("status", "staged_case_count", "unique_patients", "counts")}, indent=2))


if __name__ == "__main__":
    main()
