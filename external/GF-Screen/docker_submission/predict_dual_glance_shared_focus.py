#!/usr/bin/env python3
"""Raw-CT replay for exact bbox/CBG shared-Focus inference.

The public role retains the fixed-50 audit guard.  The test role accepts an
arbitrary challenge inventory so the same verified implementation can run in
the submission container.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import inspect
import json
import os
import platform
import sys
import time
from pathlib import Path
from types import SimpleNamespace

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

import cc3d
import nibabel as nib
import numpy as np
import torch
from monai.data import DataLoader, Dataset, decollate_batch
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
)

from inference_naming import prediction_name
from models.GF_Screen_RL import (
    GFScreen_RL,
    accumulate_segmentation_with_coords,
    finalize_segmentation_aggregation,
    focus_probability_with_tta,
)
from models.glance import Glance_model, select_crops_from_proposal
from docker_submission.winner_mask_recipe import filter_components_physical_6
from docker_submission.shared_focus_dual_glance import (
    _autocast,
    _glance_selection,
    crop_batch,
    crop_coordinates,
)


BBOX_SOURCE_SHA = "04a3cf7cb5209e65c253e6b1dd70ad7a972008e396825d5ede5ec3a933eb3fdb"
CBG_SOURCE_SHA = "3008615b57960e36b10b534b750acd4ecfec0f94950368bb80851d3734d4f190"
BBOX_COMPACT_SHA = "ba5a4ef9462bb1914f4744b1078e6251abb0f244517263e9e4b003dce4f29ad0"
CBG_COMPACT_SHA = "db8e5769afd43c9ed42bdb90cc74e906335b4ebf7b7142ab365a0c793a9ba4b1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    project = APP_ROOT.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--bbox-output", required=True, type=Path)
    parser.add_argument("--cbg-output", required=True, type=Path)
    parser.add_argument("--guided-proposal", type=Path)
    parser.add_argument("--guided-output", type=Path)
    parser.add_argument("--terminal", required=True, type=Path)
    parser.add_argument(
        "--bbox-checkpoint",
        type=Path,
        default=project / "external/GF-Screen/docker_submission/assets/gf_bbox_full_state.pt",
    )
    parser.add_argument(
        "--cbg-checkpoint",
        type=Path,
        default=project / "external/GF-Screen/docker_submission/assets/gf_cbg_glance_state.pt",
    )
    parser.add_argument("--gpu", default=0, type=int, choices=(0, 1))
    parser.add_argument("--part-id", default=0, type=int)
    parser.add_argument("--num-parts", default=1, type=int)
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--focus-batch-size", default=4, type=int)
    parser.add_argument("--glance-batch-size", default=4, type=int)
    parser.add_argument("--crop-overlap", default=0.75, type=float)
    parser.add_argument("--adaptive-overlap-selected-threshold", default=0, type=int)
    parser.add_argument(
        "--lesion-threshold", action="append", type=float, dest="lesion_thresholds",
        help="repeat to emit several Focus probability thresholds in one pass",
    )
    parser.add_argument(
        "--bbox-only",
        action="store_true",
        help="run only the bbox Glance route (deployment ablation)",
    )
    parser.add_argument("--validate-checkpoints-only", action="store_true")
    parser.add_argument("--dataset-role", required=True, choices=("public", "test"))
    return parser.parse_args()


def model_args(focus_batch_size: int) -> SimpleNamespace:
    return SimpleNamespace(
        glance_backbone="resnet18",
        grid_mode="legacy",
        glance_box_localization=False,
        glance_center_localization=False,
        in_channels=1,
        out_channels=2,
        feature_size=48,
        use_checkpoint=False,
        roi_x=96,
        roi_y=96,
        roi_z=64,
        batch_size=1,
        sw_batch_size=4,
        focus_batch_size=focus_batch_size,
        aggregation_mode="uniform",
        focus_tta_flip_dims=(),
    )


def checkpoint_payload(path: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    # ``weights_only`` was added after the challenge image's torch 1.13.1.
    # Keep one source file for both the development and deployment stacks
    # instead of silently requiring a newer PyTorch inside the submission.
    load_kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False
    payload = torch.load(path, **load_kwargs)
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"checkpoint has no state dictionary: {path}")
    return payload, state


def load_shared_models(
    bbox_checkpoint: Path,
    cbg_checkpoint: Path,
    focus_batch_size: int,
    device: torch.device,
):
    bbox_sha = sha256(bbox_checkpoint)
    cbg_sha = sha256(cbg_checkpoint)
    source_pair = (bbox_sha, cbg_sha) == (BBOX_SOURCE_SHA, CBG_SOURCE_SHA)
    compact_pair = (bbox_sha, cbg_sha) == (BBOX_COMPACT_SHA, CBG_COMPACT_SHA)
    if not source_pair and not compact_pair:
        raise RuntimeError("bbox/CBG checkpoint identity or pairing mismatch")
    bbox_payload, bbox_state = checkpoint_payload(bbox_checkpoint)
    cbg_payload, cbg_state = checkpoint_payload(cbg_checkpoint)
    focus_keys = sorted(key for key in bbox_state if key.startswith("segmentation_model."))
    if len(focus_keys) != 167:
        raise RuntimeError("bbox Focus tensor inventory mismatch")
    if source_pair:
        if focus_keys != sorted(
            key for key in cbg_state if key.startswith("segmentation_model.")
        ):
            raise RuntimeError("bbox/CBG Focus tensor inventory mismatch")
        if not all(torch.equal(bbox_state[key], cbg_state[key]) for key in focus_keys):
            raise RuntimeError("bbox/CBG Focus tensors are not bit-identical")
        cbg_glance_state = {
            key[len("glance_model.") :]: value
            for key, value in cbg_state.items()
            if key.startswith("glance_model.")
        }
        asset_mode = "source_training_checkpoints"
    else:
        if (
            bbox_payload.get("role") != "bbox_glance_and_shared_focus"
            or bbox_payload.get("source_sha256") != BBOX_SOURCE_SHA
            or cbg_payload.get("role") != "cbg_glance_only"
            or cbg_payload.get("source_sha256") != CBG_SOURCE_SHA
        ):
            raise RuntimeError("compact checkpoint provenance mismatch")
        if any(key.startswith(("glance_model.", "segmentation_model.")) for key in cbg_state):
            raise RuntimeError("compact CBG Glance keys must be module-relative")
        cbg_glance_state = cbg_state
        asset_mode = "compact_shared_focus"
    if len(cbg_glance_state) != 104:
        raise RuntimeError("CBG Glance tensor inventory mismatch")

    arguments = model_args(focus_batch_size)
    bbox = GFScreen_RL(arguments)
    bbox.load_state_dict(bbox_state, strict=True)
    cbg_glance = Glance_model(backbone="resnet18", grid_mode="legacy")
    cbg_glance.load_state_dict(cbg_glance_state, strict=True)
    del bbox_payload, cbg_payload, bbox_state, cbg_state, cbg_glance_state
    bbox.eval().to(device)
    cbg_glance.eval().to(device)
    return bbox, cbg_glance, len(focus_keys), asset_mode, bbox_sha, cbg_sha


def transforms(with_guided_proposal: bool = False):
    keys = ["image", "proposal"] if with_guided_proposal else ["image"]
    spacing_modes = ("bilinear", "nearest") if with_guided_proposal else "bilinear"
    forward = Compose(
        [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(keys=keys, pixdim=(1.0, 1.0, 3.0), mode=spacing_modes),
            ScaleIntensityRanged(
                keys=["image"], a_min=-175.0, a_max=250.0,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=keys, source_key="image"),
            SpatialPadd(keys=keys, spatial_size=(96, 96, 64), mode="constant"),
        ]
    )
    inverse = Compose(
        [
            EnsureTyped(keys=["pred"]),
            Invertd(
                keys=["pred"], transform=forward, orig_keys=["image"],
                meta_keys=["pred_meta_dict"], orig_meta_keys=["image_meta_dict"],
                meta_key_postfix="meta_dict", nearest_interp=True, to_tensor=True,
            ),
        ]
    )
    return forward, inverse


def restore_and_save(
    mask: torch.Tensor,
    batch: dict,
    inverse,
    source: Path,
    output: Path,
    *,
    minimum_mm3: float = 0.0,
):
    payload = dict(batch)
    payload["pred"] = mask.detach().cpu().to(torch.uint8).unsqueeze(1)
    restored = inverse(decollate_batch(payload)[0])["pred"]
    values = restored.detach().cpu().numpy().squeeze().astype(bool, copy=False)
    reference = nib.load(str(source))
    if values.shape != reference.shape or not np.isin(np.unique(values), [0, 1]).all():
        raise RuntimeError(f"invalid restored mask for {source.name}")
    if minimum_mm3 > 0:
        values = filter_components_physical_6(
            values,
            tuple(float(value) for value in reference.header.get_zooms()[:3]),
            minimum_mm3,
        )
    destination = output / prediction_name(source.name)
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.nii.gz")
    nib.save(nib.Nifti1Image(values, reference.affine, header), str(temporary))
    os.replace(temporary, destination)
    return destination


def infer_case(
    bbox,
    cbg_glance,
    data: torch.Tensor,
    focus_batch_size: int,
    glance_batch_size: int,
    crop_overlap: float,
    adaptive_overlap_selected_threshold: int = 0,
    guided_proposal: torch.Tensor | None = None,
    use_cbg: bool = True,
    device: torch.device | None = None,
):
    if data.shape[0] != 1:
        raise RuntimeError("deployment streaming requires batch size one")
    volume = data[0]
    target = device or next(bbox.parameters()).device
    crop_shape = (96, 96, 64)
    shape = tuple(int(value) for value in volume.shape[1:])
    def select(overlap: float):
        points = crop_coordinates(shape, crop_shape, (overlap,) * 3)
        route_a = _glance_selection(
            bbox.glance_model, volume, points, crop_shape, 0.50, glance_batch_size,
            target, True,
        )
        route_b = (
            _glance_selection(
                cbg_glance, volume, points, crop_shape, 0.20, glance_batch_size,
                target, True,
            )
            if use_cbg
            else np.zeros(len(points), dtype=bool)
        )
        return points, route_a, route_b

    coordinates, selected_a, selected_b = select(crop_overlap)
    initial_grid_crops = len(coordinates)
    initial_bbox_crops = int(selected_a.sum())
    adaptive_overlap = False
    if (
        adaptive_overlap_selected_threshold > 0
        and crop_overlap == 0.75
        and (
            initial_grid_crops > 2000
            or initial_bbox_crops > adaptive_overlap_selected_threshold
        )
    ):
        coordinates, selected_a, selected_b = select(0.625)
        adaptive_overlap = True
    proposal_selected = None
    selected_guided = None
    if guided_proposal is not None:
        if guided_proposal.shape != data.shape:
            raise RuntimeError("guided proposal/image shape mismatch")
        proposal_selected = (
            select_crops_from_proposal(
                guided_proposal[0], coordinates, min_voxels=500
            )
            .detach()
            .cpu()
            .numpy()
            .astype(bool, copy=False)
        )
        selected_guided = selected_a | proposal_selected
    selected_union = selected_a | selected_b
    if selected_guided is not None:
        selected_union |= selected_guided

    route_selected = [selected_a, selected_b]
    if selected_guided is not None:
        route_selected.append(selected_guided)
    probability = [np.zeros(shape, dtype=np.float32) for _ in route_selected]
    coverage = [np.zeros(shape, dtype=np.uint16) for _ in route_selected]
    union_ids = np.flatnonzero(selected_union).tolist()
    cx, cy, cz = crop_shape
    for start in range(0, len(union_ids), focus_batch_size):
        ids = union_ids[start : start + focus_batch_size]
        current = [coordinates[index] for index in ids]
        patches = crop_batch(volume, current, crop_shape).to(target, non_blocking=True)
        with _autocast(target, True):
            foreground = focus_probability_with_tta(
                bbox.segmentation_model, patches, flip_dims=()
            )[:, 1]
        foreground = foreground.float().detach().cpu().numpy()
        for values, index, (x, y, z) in zip(foreground, ids, current):
            for route, selected in enumerate(route_selected):
                if selected[index]:
                    probability[route][x : x + cx, y : y + cy, z : z + cz] += values
                    coverage[route][x : x + cx, y : y + cy, z : z + cz] += 1
        del patches, foreground
    stats = {
        "initial_grid_crops": initial_grid_crops,
        "initial_bbox_crops": initial_bbox_crops,
        "adaptive_overlap": adaptive_overlap,
        "grid_crops": len(coordinates),
        "bbox_crops": int(selected_a.sum()),
        "cbg_crops": int(selected_b.sum()),
        "union_crops": int(selected_union.sum()),
        "reused_crops": int(selected_a.sum() + selected_b.sum() - selected_union.sum()),
        "guided_crops": int(selected_guided.sum()) if selected_guided is not None else 0,
        "proposal_crops": int(proposal_selected.sum()) if proposal_selected is not None else 0,
    }
    if selected_guided is not None:
        stats["reused_crops"] = int(
            selected_a.sum() + selected_b.sum() + selected_guided.sum()
            - selected_union.sum()
        )
    return probability, coverage, selected_guided is not None, stats


def write_terminal(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if (args.guided_proposal is None) != (args.guided_output is None):
        raise ValueError("--guided-proposal and --guided-output must be provided together")
    if args.bbox_only and args.guided_proposal is not None:
        raise ValueError("--bbox-only cannot be combined with the guided route")
    if args.num_parts < 1 or not 0 <= args.part_id < args.num_parts:
        raise ValueError("invalid deterministic part")
    if args.workers < 0 or args.focus_batch_size < 1 or args.glance_batch_size < 1:
        raise ValueError("workers must be nonnegative and inference batch sizes positive")
    if args.adaptive_overlap_selected_threshold < 0:
        raise ValueError("adaptive overlap selected threshold must be nonnegative")
    lesion_thresholds = list(dict.fromkeys(args.lesion_thresholds or [0.50]))
    if any(not 0.0 < value < 1.0 for value in lesion_thresholds):
        raise ValueError("lesion thresholds must be in (0, 1)")
    if not 0.0 <= args.crop_overlap < 1.0:
        raise ValueError("crop overlap must be in [0, 1)")
    if not args.validate_checkpoints_only and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for shared-Focus inference")
    device = torch.device("cpu" if args.validate_checkpoints_only else f"cuda:{args.gpu}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    bbox, cbg_glance, focus_tensors, asset_mode, bbox_sha, cbg_sha = load_shared_models(
        args.bbox_checkpoint, args.cbg_checkpoint, args.focus_batch_size, device
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    initialization = time.perf_counter() - started
    initialization_peak_allocated_mb = (
        torch.cuda.max_memory_allocated(device) / (1024**2)
        if device.type == "cuda"
        else 0.0
    )
    initialization_peak_reserved_mb = (
        torch.cuda.max_memory_reserved(device) / (1024**2)
        if device.type == "cuda"
        else 0.0
    )
    common = {
        "schema_version": 1,
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "dataset_role": args.dataset_role,
        "checkpoint_validation_only": bool(args.validate_checkpoints_only),
        "checkpoint_asset_mode": asset_mode,
        "focus_tensors_verified_equal": focus_tensors,
        "model_initialization_seconds": initialization,
        "initialization_peak_gpu_allocated_mb": initialization_peak_allocated_mb,
        "initialization_peak_gpu_reserved_mb": initialization_peak_reserved_mb,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cc3d_module": str(Path(cc3d.__file__).resolve()),
        "bbox_checkpoint": {"path": str(args.bbox_checkpoint.resolve()), "sha256": bbox_sha},
        "cbg_checkpoint": {"path": str(args.cbg_checkpoint.resolve()), "sha256": cbg_sha},
        "public_labels_read": False,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
        "deployment_authorized": False,
    }
    if args.validate_checkpoints_only:
        write_terminal(args.terminal, {**common, "status": "COMPLETE", "case_count": 0, "cases": []})
        return

    names = sorted(path for path in args.images.glob("*.nii.gz") if path.is_file())
    if args.dataset_role == "public" and len(names) != 50:
        raise RuntimeError("fixed-public image inventory must contain exactly 50 cases")
    if not names:
        raise RuntimeError("image inventory is empty")
    names = names[args.part_id :: args.num_parts]
    def threshold_directory(base: Path, threshold: float) -> Path:
        if len(lesion_thresholds) == 1:
            return base
        return base / f"t{int(round(threshold * 1000)):04d}"

    bbox_outputs = {
        threshold: threshold_directory(args.bbox_output, threshold)
        for threshold in lesion_thresholds
    }
    cbg_outputs = {
        threshold: threshold_directory(args.cbg_output, threshold)
        for threshold in lesion_thresholds
    }
    for directory in (*bbox_outputs.values(), *cbg_outputs.values()):
        directory.mkdir(parents=True, exist_ok=True)
    use_guided = args.guided_proposal is not None
    if use_guided:
        expected_proposals = {prediction_name(path.name) for path in names}
        observed_proposals = {path.name for path in args.guided_proposal.glob("*.nii.gz")}
        if not expected_proposals.issubset(observed_proposals):
            raise RuntimeError("guided proposal inventory mismatch")
        guided_outputs = {
            threshold: threshold_directory(args.guided_output, threshold)
            for threshold in lesion_thresholds
        }
        for directory in guided_outputs.values():
            directory.mkdir(parents=True, exist_ok=True)
    else:
        guided_outputs = {}
    forward, inverse = transforms(use_guided)
    data_items = []
    for path in names:
        item = {"image": str(path), "source": str(path)}
        if use_guided:
            item["proposal"] = str(args.guided_proposal / prediction_name(path.name))
        data_items.append(item)
    loader = DataLoader(
        Dataset(
            data=data_items,
            transform=forward,
        ),
        batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True,
    )
    # The evaluation GPU has 16 GiB. Autotuning on this 48-GiB development GPU
    # otherwise selects convolution workspaces that cannot exist at test time.
    torch.backends.cudnn.benchmark = False
    rows = []
    for batch in loader:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        case_started = time.perf_counter()
        source = Path(batch["source"][0])
        # Keep the resampled full CT on host RAM; only micro-batched 96x96x64
        # crops enter CUDA. This is required for the official 16-GB GPU.
        data = batch["image"]
        proposal = batch.get("proposal")
        # Proposal crop selection is an integral-image operation. Keep it on
        # CPU to avoid a transient full-volume int32 CUDA allocation.
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True):
            probability, coverage, has_guided, stats = infer_case(
                bbox,
                cbg_glance,
                data,
                args.focus_batch_size,
                args.glance_batch_size,
                args.crop_overlap,
                args.adaptive_overlap_selected_threshold,
                proposal,
                use_cbg=not args.bbox_only,
                device=device,
            )
        for threshold in lesion_thresholds:
            masks = [
                torch.from_numpy(
                    (count > 0) & (total >= threshold * count)
                ).unsqueeze(0)
                for total, count in zip(probability, coverage)
            ]
            restore_and_save(masks[0], batch, inverse, source, bbox_outputs[threshold])
            restore_and_save(masks[1], batch, inverse, source, cbg_outputs[threshold])
            if has_guided:
                restore_and_save(
                    masks[2], batch, inverse, source, guided_outputs[threshold],
                    minimum_mm3=500.0,
                )
            del masks
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - case_started
        rows.append(
            {
                "case": prediction_name(source.name),
                "wall_seconds": elapsed,
                "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated(device)
                / (1024**2),
                "peak_gpu_reserved_mb": torch.cuda.max_memory_reserved(device)
                / (1024**2),
                **stats,
            }
        )
        print(json.dumps(rows[-1], sort_keys=True), flush=True)
        del data, probability, coverage
        if proposal is not None:
            del proposal
        torch.cuda.empty_cache()
    write_terminal(
        args.terminal,
        {
            **common,
            "status": "COMPLETE",
            "case_count": len(rows),
            "part_id": args.part_id,
            "num_parts": args.num_parts,
            "focus_batch_size": args.focus_batch_size,
            "glance_batch_size": args.glance_batch_size,
            "crop_overlap": args.crop_overlap,
            "adaptive_overlap_selected_threshold": args.adaptive_overlap_selected_threshold,
            "lesion_thresholds": lesion_thresholds,
            "guided_route": bool(use_guided),
            "bbox_only": bool(args.bbox_only),
            "guided_proposal_min_voxels": 500 if use_guided else None,
            "guided_native_filter_mm3": 500.0 if use_guided else None,
            "wall_seconds": time.perf_counter() - started,
            "peak_gpu_allocated_mb": max(
                [initialization_peak_allocated_mb]
                + [row["peak_gpu_allocated_mb"] for row in rows]
            ),
            "peak_gpu_reserved_mb": max(
                [initialization_peak_reserved_mb]
                + [row["peak_gpu_reserved_mb"] for row in rows]
            ),
            "cases": rows,
        },
    )


if __name__ == "__main__":
    main()
