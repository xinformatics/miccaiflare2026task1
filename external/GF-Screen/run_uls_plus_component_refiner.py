#!/usr/bin/env python3
"""Use automatic candidate components as click prompts for the ULS+ model.

ULS+ is an interactive, click-centred lesion segmenter.  This adapter keeps
detection and segmentation separate: connected components from an automatic
proposal mask supply deterministic interior click points, ULS+ refines one
128x128x64 VOI per click, and the refined VOIs are mapped back to the source
geometry.  No ground-truth mask is read by this program.

The three stages are independently resumable.  In particular, ``infer`` is
the only GPU stage; ``materialize`` can be repeated with different confirmation
thresholds without rerunning ULS+.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from scipy import ndimage


FILE_ENDING = ".nii.gz"
CROP_SIZE_XYZ = (128, 128, 64)


def strip_nii(path: Path) -> str:
    name = path.name
    return name[:-7] if name.endswith(FILE_ENDING) else path.stem


def image_for_case(images: Path, case: str) -> Path:
    choices = (images / f"{case}{FILE_ENDING}", images / f"{case}_0000{FILE_ENDING}")
    found = [p for p in choices if p.is_file()]
    if len(found) != 1:
        raise RuntimeError(f"expected one CT for {case}, found: {found}")
    return found[0]


def same_geometry(a: sitk.Image, b: sitk.Image, atol: float = 1e-4) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=atol, rtol=0)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=atol, rtol=0)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=atol, rtol=0)
    )


def load_official_cropper(uls_root: Path):
    preprocessing = str((uls_root / "preprocessing").resolve())
    if preprocessing not in sys.path:
        sys.path.insert(0, preprocessing)
    from uclp.cropper import compute_crop_region, extract_crop  # type: ignore

    return compute_crop_region, extract_crop


def interior_voxel(component: np.ndarray, spacing_zyx: tuple[float, float, float]) -> tuple[int, int, int]:
    """Return a deterministic, maximally interior (z, y, x) click voxel."""
    distance = ndimage.distance_transform_edt(component, sampling=spacing_zyx)
    flat = int(np.argmax(distance))
    return tuple(int(v) for v in np.unravel_index(flat, component.shape))


def prepare(args: argparse.Namespace) -> list[dict[str, Any]]:
    args.crops.mkdir(parents=True, exist_ok=True)
    compute_crop_region, extract_crop = load_official_cropper(args.uls_root)
    proposal_files = sorted(args.proposals.glob(f"*{FILE_ENDING}"))
    if args.case_id:
        requested = set(args.case_id)
        proposal_files = [p for p in proposal_files if strip_nii(p) in requested]
        found = {strip_nii(p) for p in proposal_files}
        if found != requested:
            raise RuntimeError(f"requested cases not found: {sorted(requested - found)}")
    if args.case_limit:
        proposal_files = proposal_files[: args.case_limit]
    if not proposal_files:
        raise RuntimeError(f"no proposals found in {args.proposals}")

    records: list[dict[str, Any]] = []
    structure = ndimage.generate_binary_structure(3, 3)
    for proposal_path in proposal_files:
        case = strip_nii(proposal_path)
        ct_path = image_for_case(args.images, case)
        ct = sitk.ReadImage(str(ct_path))
        proposal = sitk.ReadImage(str(proposal_path))
        if not same_geometry(ct, proposal):
            raise RuntimeError(f"geometry mismatch for {case}")
        mask = sitk.GetArrayFromImage(proposal) > 0
        labels, count = ndimage.label(mask, structure=structure)
        component_slices = ndimage.find_objects(labels)
        component_sizes = np.bincount(labels.ravel(), minlength=count + 1)
        spacing_xyz = tuple(float(v) for v in ct.GetSpacing())
        spacing_zyx = tuple(reversed(spacing_xyz))
        voxel_mm3 = float(np.prod(spacing_xyz))
        candidates: list[tuple[int, int]] = []
        for label_id in range(1, count + 1):
            voxels = int(component_sizes[label_id])
            volume = voxels * voxel_mm3
            if args.min_seed_mm3 <= volume <= args.max_seed_mm3:
                candidates.append((label_id, voxels))
        candidates.sort(key=lambda item: (-item[1], item[0]))
        candidates = candidates[: args.max_components_per_case]

        for rank, (label_id, seed_voxels) in enumerate(candidates):
            bounds = component_slices[label_id - 1]
            if bounds is None:
                raise RuntimeError(f"missing component bounds for {case} label {label_id}")
            component = labels[bounds] == label_id
            local_zyx = interior_voxel(component, spacing_zyx)
            z, y, x = tuple(
                int(local + axis.start) for local, axis in zip(local_zyx, bounds)
            )
            point = ct.TransformIndexToPhysicalPoint((x, y, z))
            start_xyz, _ = compute_crop_region(
                point, CROP_SIZE_XYZ, ct.GetSize(), ct.GetSpacing(), image=ct
            )
            crop = extract_crop(ct, (start_xyz, CROP_SIZE_XYZ), is_mask=False)
            crop_id = f"{case}__c{rank:04d}"
            crop_path = args.crops / f"{crop_id}_0000{FILE_ENDING}"
            if not crop_path.is_file():
                sitk.WriteImage(crop, str(crop_path), True)
            records.append(
                {
                    "case": case,
                    "image": str(ct_path.resolve()),
                    "proposal": str(proposal_path.resolve()),
                    "crop_id": crop_id,
                    "crop": str(crop_path.resolve()),
                    "label_id": label_id,
                    "rank": rank,
                    "seed_voxels": seed_voxels,
                    "seed_mm3": seed_voxels * voxel_mm3,
                    "click_zyx": [z, y, x],
                    "start_xyz": [int(v) for v in start_xyz],
                    "crop_size_xyz": list(CROP_SIZE_XYZ),
                }
            )
        print(f"prepared {case}: {len(candidates)} components", flush=True)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(records, indent=2) + "\n")
    print(f"prepared {len(records)} VOIs from {len(proposal_files)} cases", flush=True)
    return records


def patch_identity_resampler(uls_root: Path) -> None:
    path = uls_root / "nnunetv2/preprocessing/resampling/custom_resampling.py"
    spec = importlib.util.spec_from_file_location("uls_plus_custom_resampling", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import ULS+ resampler from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import nnunetv2.utilities.plans_handling.plans_handler as plans_handler

    original = plans_handler.recursive_find_resampling_fn_by_name

    def resolve(name: str):
        if name == "no_resampling_data_or_seg_to_shape":
            return module.no_resampling_data_or_seg_to_shape
        return original(name)

    plans_handler.recursive_find_resampling_fn_by_name = resolve


def infer(args: argparse.Namespace, records: list[dict[str, Any]]) -> None:
    if not records:
        raise RuntimeError("manifest contains no candidate VOIs")
    os.environ.setdefault("MPLCONFIGDIR", str(args.work_dir / "matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    # nnU-Net preprocesses files in spawned Python interpreters. Put the local
    # startup shim on PYTHONPATH so each child resolves the released identity
    # resampler without editing the shared site-packages installation.
    overlay = str((args.uls_root / "python_overlay").resolve())
    inherited = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = overlay if not inherited else overlay + os.pathsep + inherited
    patch_identity_resampler(args.uls_root)
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    device = torch.device(args.device)
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=not args.disable_mirroring,
        perform_everything_on_device=device.type == "cuda",
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        str(args.model), use_folds=("all",), checkpoint_name=args.checkpoint
    )
    args.predicted_crops.mkdir(parents=True, exist_ok=True)
    inputs = [[r["crop"]] for r in records]
    outputs = [str((args.predicted_crops / r["crop_id"]).resolve()) for r in records]
    predictor.predict_from_files(
        inputs,
        outputs,
        save_probabilities=args.save_probabilities,
        overwrite=False,
        num_processes_preprocessing=args.preprocessing_workers,
        num_processes_segmentation_export=args.export_workers,
    )


def paste_crop(crop: np.ndarray, full_shape: tuple[int, int, int], start_xyz: list[int]) -> np.ndarray:
    """Paste a z-y-x crop into full z-y-x geometry, clipping padded margins."""
    start_zyx = np.asarray(tuple(reversed(start_xyz)), dtype=int)
    full = np.zeros(full_shape, dtype=bool)
    dst_lo = np.maximum(start_zyx, 0)
    src_lo = np.maximum(-start_zyx, 0)
    length = np.minimum(np.asarray(full_shape) - dst_lo, np.asarray(crop.shape) - src_lo)
    if np.any(length <= 0):
        return full
    dst = tuple(slice(int(lo), int(lo + n)) for lo, n in zip(dst_lo, length))
    src = tuple(slice(int(lo), int(lo + n)) for lo, n in zip(src_lo, length))
    full[dst] = crop[src] > 0
    return full


def materialize(args: argparse.Namespace, records: list[dict[str, Any]]) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    by_case: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_case.setdefault(record["case"], []).append(record)
    proposal_files = sorted(args.proposals.glob(f"*{FILE_ENDING}"))
    if args.case_id:
        requested = set(args.case_id)
        proposal_files = [p for p in proposal_files if strip_nii(p) in requested]
    if args.case_limit:
        proposal_files = proposal_files[: args.case_limit]
    if not proposal_files:
        raise RuntimeError(f"no proposals found in {args.proposals}")
    rows: list[dict[str, Any]] = []
    for proposal_path in proposal_files:
        case = strip_nii(proposal_path)
        case_records = by_case.get(case, [])
        proposal_img = sitk.ReadImage(str(proposal_path))
        proposal = sitk.GetArrayFromImage(proposal_img) > 0
        seed_labels, _ = ndimage.label(proposal, structure=ndimage.generate_binary_structure(3, 3))
        fused = np.zeros(proposal.shape, dtype=bool)
        for record in case_records:
            pred_path = args.predicted_crops / f"{record['crop_id']}{FILE_ENDING}"
            if not pred_path.is_file():
                raise RuntimeError(f"missing ULS+ prediction: {pred_path}")
            pred_crop = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path))) > 0
            refined = paste_crop(pred_crop, proposal.shape, record["start_xyz"])
            seed = seed_labels == int(record["label_id"])
            intersection = int(np.count_nonzero(refined & seed))
            refined_voxels = int(np.count_nonzero(refined))
            seed_voxels = int(np.count_nonzero(seed))
            seed_coverage = intersection / max(seed_voxels, 1)
            refined_overlap = intersection / max(refined_voxels, 1)
            volume_ratio = refined_voxels / max(seed_voxels, 1)
            accepted = (
                refined_voxels > 0
                and seed_coverage >= args.min_seed_coverage
                and refined_overlap >= args.min_refined_overlap
                and args.min_volume_ratio <= volume_ratio <= args.max_volume_ratio
            )
            if accepted:
                fused |= refined
            rows.append(
                {
                    "case": case,
                    "crop_id": record["crop_id"],
                    "seed_voxels": seed_voxels,
                    "refined_voxels": refined_voxels,
                    "intersection_voxels": intersection,
                    "seed_coverage": seed_coverage,
                    "refined_overlap": refined_overlap,
                    "volume_ratio": volume_ratio,
                    "accepted": int(accepted),
                }
            )
        if args.base is not None:
            base_path = args.base / f"{case}{FILE_ENDING}"
            if not base_path.is_file():
                raise RuntimeError(f"missing base mask: {base_path}")
            base_img = sitk.ReadImage(str(base_path))
            if not same_geometry(base_img, proposal_img):
                raise RuntimeError(f"base geometry mismatch for {case}")
            fused |= sitk.GetArrayFromImage(base_img) > 0
        output_img = sitk.GetImageFromArray(fused.astype(np.uint8))
        output_img.CopyInformation(proposal_img)
        sitk.WriteImage(output_img, str(args.output / f"{case}{FILE_ENDING}"), True)
        print(f"materialized {case}", flush=True)

    stats = args.output / "component_stats.csv"
    with stats.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["case"])
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "ULS_PLUS_COMPONENT_REFINEMENT_COMPLETE.json").write_text(
        json.dumps(
            {
                "cases": len(proposal_files),
                "components": len(rows),
                "accepted": sum(int(r["accepted"]) for r in rows),
                "min_seed_coverage": args.min_seed_coverage,
                "min_refined_overlap": args.min_refined_overlap,
                "min_volume_ratio": args.min_volume_ratio,
                "max_volume_ratio": args.max_volume_ratio,
                "base": str(args.base.resolve()) if args.base else None,
            },
            indent=2,
        )
        + "\n"
    )


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[2]
    uls_root = project / "external/ULS-Plus"
    model = uls_root / "weights/nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres"
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "prepare", "infer", "materialize"), default="all")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--uls-root", type=Path, default=uls_root)
    parser.add_argument("--model", type=Path, default=model)
    parser.add_argument("--checkpoint", default="checkpoint_best.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--disable-mirroring", action="store_true")
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument("--preprocessing-workers", type=int, default=2)
    parser.add_argument("--export-workers", type=int, default=2)
    parser.add_argument("--min-seed-mm3", type=float, default=50.0)
    parser.add_argument("--max-seed-mm3", type=float, default=200000.0)
    parser.add_argument("--max-components-per-case", type=int, default=32)
    parser.add_argument("--min-seed-coverage", type=float, default=0.05)
    parser.add_argument("--min-refined-overlap", type=float, default=0.05)
    parser.add_argument("--min-volume-ratio", type=float, default=0.1)
    parser.add_argument("--max-volume-ratio", type=float, default=10.0)
    parser.add_argument("--case-limit", type=int, default=0)
    parser.add_argument("--case-id", action="append", help="prepare only this exact case id; repeatable")
    args = parser.parse_args()
    args.uls_root = args.uls_root.resolve()
    args.model = args.model.resolve()
    args.work_dir = args.work_dir.resolve()
    args.crops = args.work_dir / "crops"
    args.predicted_crops = args.work_dir / "predicted_crops"
    args.manifest = args.work_dir / "manifest.json"
    return args


def main() -> None:
    args = parse_args()
    required = [
        args.images,
        args.proposals,
        args.uls_root,
        args.model / "fold_all" / args.checkpoint,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"missing required paths: {missing}")
    records: list[dict[str, Any]]
    if args.stage in ("all", "prepare"):
        records = prepare(args)
    else:
        records = json.loads(args.manifest.read_text())
    if args.stage in ("all", "infer"):
        infer(args, records)
    if args.stage in ("all", "materialize"):
        materialize(args, records)


if __name__ == "__main__":
    main()
