#!/usr/bin/env python3
"""Fast paired Focus checkpoint screen on deterministic lesion components.

Full GF validation repeatedly runs Glance over every crop in a volume, even
when only Focus tensors differ between checkpoints.  This diagnostic instead
builds one deterministic, target-centred crop for representative annotated
components and evaluates every candidate on the identical tensors.  Metrics
are restricted to an expanded component box, so unannotated disease elsewhere
in a partially labelled scan is never treated as a false positive.

This is an internal screening gate, not a replacement for complete validation
or the fixed-public challenge gate.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from GF_RL_train import parser as training_parser
from models.GF_Screen_RL import GFScreen_RL
from utils.data_utils import get_abdomen_ds, get_chest_ds


ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = Path(
    "/mnt/nvme3-data/shashank/MICCAI26/gf_screen_preprocessed_1x1x3"
)
DEFAULT_ROI = (96, 96, 64)
SPACING = (1.0, 1.0, 3.0)
CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=np.uint8)


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must have LABEL=PATH form")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("checkpoint must have LABEL=PATH form")
    return label, Path(raw_path)


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def stable_key(seed: int, name: str) -> bytes:
    return hashlib.sha256(f"{seed}:{name}".encode()).digest()


def source_items(dataset) -> list[dict]:
    items = getattr(dataset, "data", None)
    if not isinstance(items, list):
        raise TypeError(f"dataset has no list-valued source inventory: {type(dataset)}")
    return items


def select_indices(dataset, count: int, seed: int, lct: bool) -> list[int]:
    candidates = []
    for index, item in enumerate(source_items(dataset)):
        name = str(item.get("name", ""))
        if name.startswith("WholeBody_LCT_") == lct:
            candidates.append((stable_key(seed, name), index))
    candidates.sort()
    return [index for _key, index in candidates[:count]]


def component_ids_by_scale(component_map: np.ndarray, maximum: int) -> list[int]:
    counts = np.bincount(component_map.reshape(-1))
    ranked = sorted(
        ((int(counts[index]), index) for index in range(1, len(counts))),
        key=lambda pair: (pair[0], pair[1]),
    )
    if len(ranked) <= maximum:
        return [component_id for _size, component_id in ranked]
    positions = np.linspace(0, len(ranked) - 1, maximum)
    selected = []
    for position in positions:
        component_id = ranked[int(round(float(position)))][1]
        if component_id not in selected:
            selected.append(component_id)
    return selected


def crop_slices(
    center: np.ndarray,
    shape: tuple[int, int, int],
    roi: tuple[int, int, int] = DEFAULT_ROI,
) -> tuple[slice, ...]:
    result = []
    for coordinate, size, extent in zip(center, roi, shape):
        if extent < size:
            raise ValueError(f"preprocessed shape {shape} is smaller than ROI {roi}")
        start = int(round(float(coordinate))) - size // 2
        start = max(0, min(start, extent - size))
        result.append(slice(start, start + size))
    return tuple(result)


def component_box(component: np.ndarray, context_mm: float) -> np.ndarray:
    coordinates = np.argwhere(component)
    if not len(coordinates):
        raise ValueError("empty component crop")
    margins = np.ceil(float(context_mm) / np.asarray(SPACING)).astype(int)
    lower = np.maximum(0, coordinates.min(0) - margins)
    upper = np.minimum(component.shape, coordinates.max(0) + 1 + margins)
    box = np.zeros(component.shape, dtype=bool)
    box[tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))] = True
    return box


def prepare_crops(
    datasets: list[tuple[str, object]],
    cases_per_group: int,
    components_per_case: int,
    seed: int,
    context_mm: float,
    roi: tuple[int, int, int] = DEFAULT_ROI,
) -> list[dict]:
    crops = []
    for region_index, (region, dataset) in enumerate(datasets):
        for lct in (False, True):
            indices = select_indices(
                dataset,
                count=cases_per_group,
                seed=seed + 1009 * region_index + 97 * int(lct),
                lct=lct,
            )
            if len(indices) < cases_per_group:
                raise ValueError(
                    f"only {len(indices)} cases available for region={region} lct={lct}"
                )
            for dataset_index in indices:
                item = dataset[dataset_index]
                image = torch.as_tensor(item["image"]).float().cpu()
                label = torch.as_tensor(item["label"])[0].gt(0.5).cpu().numpy()
                name = str(item.get("name", source_items(dataset)[dataset_index]["name"]))
                component_map, component_count = ndimage.label(label, CONNECTIVITY_26)
                if component_count < 1:
                    continue
                for component_id in component_ids_by_scale(
                    component_map, components_per_case
                ):
                    native_component = component_map == component_id
                    center = np.argwhere(native_component).mean(0)
                    slices = crop_slices(center, tuple(label.shape), roi=roi)
                    component = native_component[slices]
                    evaluation_box = component_box(component, context_mm)
                    crops.append(
                        {
                            "name": name,
                            "region": region,
                            "cohort": "lct" if lct else "original",
                            "component_id": int(component_id),
                            "component_voxels": int(component.sum()),
                            "component_mm3": float(component.sum() * np.prod(SPACING)),
                            "image": image[(slice(None), *slices)].contiguous(),
                            "target": component,
                            "evaluation_box": evaluation_box,
                        }
                    )
    if not crops:
        raise RuntimeError("deterministic component screen produced no crops")
    return crops


def focus_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = ("segmentation_model.", "module.segmentation_model.")
    result = {}
    for key, value in state_dict.items():
        for prefix in prefixes:
            if key.startswith(prefix):
                result[key[len(prefix) :]] = value
                break
    if not result:
        raise ValueError("checkpoint contains no Focus tensors")
    return result


def evaluate_checkpoint(
    label: str,
    path: Path,
    crops: list[dict],
    args: argparse.Namespace,
    batch_size: int,
) -> tuple[dict, list[dict]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    full_model = GFScreen_RL(args)
    focus = full_model.segmentation_model
    focus.load_state_dict(focus_state(checkpoint["state_dict"]), strict=True)
    del full_model, checkpoint
    focus.cuda(args.gpu).eval()

    rows = []
    with torch.inference_mode():
        for start in range(0, len(crops), batch_size):
            batch = crops[start : start + batch_size]
            images = torch.stack([item["image"] for item in batch]).cuda(
                args.gpu, non_blocking=True
            )
            with torch.autocast(device_type="cuda", enabled=bool(args.amp)):
                prediction = focus(images).argmax(1).cpu().numpy().astype(bool)
            for item, predicted in zip(batch, prediction):
                target = item["target"]
                box = item["evaluation_box"]
                predicted = predicted & box
                tp = int(np.count_nonzero(predicted & target))
                pred_count = int(np.count_nonzero(predicted))
                target_count = int(np.count_nonzero(target))
                denominator = pred_count + target_count
                rows.append(
                    {
                        "checkpoint": label,
                        "name": item["name"],
                        "region": item["region"],
                        "cohort": item["cohort"],
                        "component_id": item["component_id"],
                        "component_voxels": item["component_voxels"],
                        "component_mm3": item["component_mm3"],
                        "tp": tp,
                        "prediction_voxels_in_box": pred_count,
                        "dice": (2.0 * tp / denominator) if denominator else 1.0,
                        "recall": tp / max(1, target_count),
                        "precision": tp / max(1, pred_count),
                        "detected_at_10pct": tp / max(1, target_count) >= 0.10,
                    }
                )
    del focus
    torch.cuda.empty_cache()

    def aggregate(subset: list[dict]) -> dict:
        tp = sum(row["tp"] for row in subset)
        predicted = sum(row["prediction_voxels_in_box"] for row in subset)
        target = sum(row["component_voxels"] for row in subset)
        return {
            "components": len(subset),
            "cases": len({row["name"] for row in subset}),
            "macro_component_dice": float(np.mean([row["dice"] for row in subset])),
            "micro_component_dice": 2.0 * tp / max(1, predicted + target),
            "voxel_recall": tp / max(1, target),
            "voxel_precision": tp / max(1, predicted),
            "component_recall_at_10pct": float(
                np.mean([row["detected_at_10pct"] for row in subset])
            ),
        }

    groups = {"all": aggregate(rows)}
    for cohort in ("original", "lct"):
        subset = [row for row in rows if row["cohort"] == cohort]
        groups[cohort] = aggregate(subset)
        for region in ("abdomen", "chest"):
            region_subset = [row for row in subset if row["region"] == region]
            groups[f"{cohort}_{region}"] = aggregate(region_subset)
    return {
        "label": label,
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "metrics": groups,
    }, rows


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", action="append", type=parse_checkpoint, required=True
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--cases-per-group", default=8, type=int)
    parser.add_argument("--components-per-case", default=3, type=int)
    parser.add_argument("--context-mm", default=16.0, type=float)
    parser.add_argument("--roi-x", default=DEFAULT_ROI[0], type=int)
    parser.add_argument("--roi-y", default=DEFAULT_ROI[1], type=int)
    parser.add_argument("--roi-z", default=DEFAULT_ROI[2], type=int)
    parser.add_argument("--seed", default=20260831, type=int)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE, type=Path)
    parser.add_argument(
        "--abdomen-json",
        default=ROOT / "jsons/lct_task2_20260830/abdomen.json",
        type=Path,
    )
    parser.add_argument(
        "--chest-json",
        default=ROOT / "jsons/lct_task2_20260830/chest.json",
        type=Path,
    )
    return parser


def main() -> None:
    cli = build_cli().parse_args()
    if cli.output.exists():
        raise FileExistsError(f"refusing to overwrite {cli.output}")
    if min(cli.batch_size, cli.cases_per_group, cli.components_per_case) < 1:
        raise ValueError("batch/case/component counts must be positive")
    if cli.context_mm < 0:
        raise ValueError("context must be non-negative")
    roi = (cli.roi_x, cli.roi_y, cli.roi_z)
    if any(size < 32 or size % 32 for size in roi):
        raise ValueError("ROI dimensions must be positive multiples of 32")
    checkpoint_labels = [label for label, _path in cli.checkpoint]
    if len(checkpoint_labels) != len(set(checkpoint_labels)):
        raise ValueError("checkpoint labels must be unique")
    for _label, path in cli.checkpoint:
        if not path.is_file():
            raise FileNotFoundError(path)

    args = training_parser.parse_args([])
    args.data_dir = str(ROOT)
    args.cache_dir = str(cli.cache_dir.resolve())
    args.abdomen_json = str(cli.abdomen_json.resolve())
    args.chest_json = str(cli.chest_json.resolve())
    args.workers = 0
    args.gpu = cli.gpu
    args.rank = cli.gpu
    args.roi_x, args.roi_y, args.roi_z = roi
    args.amp = not bool(args.noamp)
    args.use_persistent_dataset = True

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(cli.gpu)
    abdomen_validation = get_abdomen_ds(args)[1]
    chest_validation = get_chest_ds(args)[1]
    crops = prepare_crops(
        [("abdomen", abdomen_validation), ("chest", chest_validation)],
        cases_per_group=cli.cases_per_group,
        components_per_case=cli.components_per_case,
        seed=cli.seed,
        context_mm=cli.context_mm,
        roi=roi,
    )

    checkpoint_results = []
    all_rows = []
    for label, path in cli.checkpoint:
        result, rows = evaluate_checkpoint(
            label, path, crops, args, batch_size=cli.batch_size
        )
        checkpoint_results.append(result)
        all_rows.extend(rows)
        metrics = result["metrics"]
        print(
            f"FOCUS_SCREEN {label} "
            f"original={metrics['original']['macro_component_dice']:.6f} "
            f"lct={metrics['lct']['macro_component_dice']:.6f} "
            f"all={metrics['all']['macro_component_dice']:.6f}",
            flush=True,
        )

    payload = {
        "schema_version": 1,
        "created_at": dt.datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "role": "internal_paired_focus_screen_not_checkpoint_promotion",
        "selection": {
            "seed": cli.seed,
            "cases_per_region_and_cohort": cli.cases_per_group,
            "maximum_components_per_case": cli.components_per_case,
            "component_selection": "size-stratified_small_median_large",
            "roi": list(roi),
            "spacing_mm": list(SPACING),
            "evaluation_box_context_mm": cli.context_mm,
            "crop_count": len(crops),
            "case_count": len({item["name"] for item in crops}),
        },
        "checkpoints": checkpoint_results,
        "component_rows": all_rows,
        "limitations": [
            "target-centred Focus crops do not measure Glance proposal recall",
            "partial annotations make prediction-only disease outside each target box unknown",
            "promotion still requires complete internal and fixed-public evaluation",
        ],
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    atomic_json(cli.output, payload)


if __name__ == "__main__":
    main()
