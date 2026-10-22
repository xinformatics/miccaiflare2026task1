#!/usr/bin/env python3
"""Run the two frozen winner prompts with one persistent BiomedParse model.

The winner used two *separate* forward passes, one for ``whole-body lesion``
and one for ``Tumor``.  Joining the text prompts changes model semantics, so
this runner only removes duplicate checkpoint loading; it deliberately calls
``predict_case`` twice and writes the same per-prompt NPZ contract as the
historical single-prompt runner.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


PROMPT_ROUTES = (
    ("whole_body_lesion", "whole-body lesion"),
    ("tumor", "Tumor"),
)
COMPACT_CHECKPOINT_SHA256 = (
    "ef4b09b78d00ce1409e81781b51f7b9968b5043bc268d6df6c9755b9a540526b"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def threshold_key(threshold: float) -> str:
    return f"segs_e{int(round(threshold * 100)):03d}"


def output_fields(
    segmentations: dict[float, np.ndarray],
    existence: np.ndarray,
    elapsed: float,
    prompt: str,
    thresholds: list[float],
) -> dict[str, np.ndarray]:
    fields = {
        threshold_key(threshold): segmentations[threshold]
        for threshold in thresholds
    }
    fields["segs"] = segmentations[thresholds[0]]
    fields.update(
        object_existence=existence,
        existence_thresholds=np.asarray(thresholds, dtype=np.float32),
        inference_seconds=np.asarray(elapsed, dtype=np.float32),
        prompts=np.asarray([prompt]),
    )
    return fields


def atomic_save_npz(destination: Path, fields: dict[str, np.ndarray]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial.npz")
    np.savez_compressed(temporary, **fields)
    os.replace(temporary, destination)


def output_complete(path: Path, thresholds: list[float], prompt: str) -> bool:
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = [threshold_key(value) for value in thresholds]
            required += ["segs", "object_existence", "existence_thresholds", "prompts"]
            if not all(key in archive for key in required):
                return False
            return list(np.asarray(archive["prompts"]).astype(str)) == [prompt]
    except (OSError, ValueError, KeyError):
        return False


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    project = root.parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--whole-output-dir", type=Path, required=True)
    parser.add_argument("--tumor-output-dir", type=Path, required=True)
    parser.add_argument("--terminal", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project
        / "external/GF-Screen/docker_submission/assets/biomedparse_3d_state_only.pt",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--slice-batch-size", type=int, default=4)
    parser.add_argument(
        "--existence-threshold",
        type=float,
        action="append",
        dest="existence_thresholds",
        help="repeat to preserve multiple historical operating points in one pass",
    )
    parser.add_argument("--continue-prediction", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--dataset-role", required=True, choices=("public", "test"))
    parser.add_argument("--validate-checkpoint-only", action="store_true")
    return parser.parse_args()


def write_terminal(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.slice_batch_size < 1:
        raise ValueError("slice batch size must be positive")
    thresholds = args.existence_thresholds or [0.50]
    thresholds = list(dict.fromkeys(thresholds))
    if any(not 0.0 <= threshold <= 1.0 for threshold in thresholds):
        raise ValueError("existence threshold must be in [0,1]")
    if len({threshold_key(value) for value in thresholds}) != len(thresholds):
        raise ValueError("existence thresholds must differ by at least 0.01")
    if 0.50 not in thresholds:
        raise ValueError("the frozen winner contract requires existence threshold 0.50")
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("invalid deterministic shard")
    checkpoint_hash = sha256(args.checkpoint)
    if checkpoint_hash != COMPACT_CHECKPOINT_SHA256:
        raise RuntimeError("compact BiomedParse checkpoint SHA-256 mismatch")

    import torch

    from miccai_inference import load_model, predict_case

    device = torch.device("cpu" if args.validate_checkpoint_only else args.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    initialization_started = time.perf_counter()
    model = load_model(device, checkpoint=args.checkpoint)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    initialization_seconds = time.perf_counter() - initialization_started
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
    import detectron2

    common = {
        "schema_version": 1,
        "status": "CHECKPOINT_VALIDATED" if args.validate_checkpoint_only else "RUNNING",
        "dataset_role": args.dataset_role,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "model_loads": 1,
        "prompt_execution": "two_separate_forward_passes",
        "prompt_order": [prompt for _, prompt in PROMPT_ROUTES],
        "initialization_seconds": initialization_seconds,
        "initialization_peak_gpu_allocated_mb": initialization_peak_allocated_mb,
        "initialization_peak_gpu_reserved_mb": initialization_peak_reserved_mb,
        "torch_version": torch.__version__,
        "detectron2_module": str(Path(detectron2.__file__).resolve()),
        "public_labels_read": False,
        "hidden_read": args.dataset_role == "test",
        "submission_created": False,
        "uploaded": False,
    }
    if args.validate_checkpoint_only:
        write_terminal(args.terminal, common)
        print(json.dumps(common, indent=2), flush=True)
        return

    files = sorted(args.input_dir.glob("*.npz"))[args.shard_id :: args.num_shards]
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"no NPZ files in {args.input_dir}")
    outputs = {
        "whole_body_lesion": args.whole_output_dir,
        "tumor": args.tumor_output_dir,
    }
    for directory in outputs.values():
        directory.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, source in enumerate(files, 1):
        destinations = {
            route: outputs[route] / source.name for route, _ in PROMPT_ROUTES
        }
        if args.continue_prediction and all(
            output_complete(destinations[route], thresholds, prompt)
            for route, prompt in PROMPT_ROUTES
        ):
            print(f"skipped {index}/{len(files)} {source.name}: complete", flush=True)
            continue
        with np.load(source, allow_pickle=False) as archive:
            image = np.asarray(archive["imgs"])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        case_started = time.perf_counter()
        route_row = {"case": source.name}
        for route, prompt in PROMPT_ROUTES:
            segmentations, existence, elapsed = predict_case(
                model,
                image,
                [prompt],
                device,
                args.slice_batch_size,
                thresholds,
                True,
            )
            atomic_save_npz(
                destinations[route],
                output_fields(segmentations, existence, elapsed, prompt, thresholds),
            )
            route_row[f"{route}_seconds"] = elapsed
            route_row[f"{route}_foreground"] = int(segmentations[0.50].sum())
            del segmentations, existence
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        route_row["wall_seconds"] = time.perf_counter() - case_started
        route_row["peak_gpu_allocated_mb"] = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else 0.0
        )
        route_row["peak_gpu_reserved_mb"] = (
            torch.cuda.max_memory_reserved(device) / (1024**2)
            if device.type == "cuda"
            else 0.0
        )
        rows.append(route_row)
        print(
            f"predicted {index}/{len(files)} {source.name}: "
            f"whole={route_row['whole_body_lesion_foreground']} "
            f"tumor={route_row['tumor_foreground']} "
            f"wall={route_row['wall_seconds']:.2f}s",
            flush=True,
        )
        del image
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    terminal = {**common, **{
        "status": "COMPLETE",
        "case_count": len(files),
        "predicted_case_count": len(rows),
        "existence_thresholds": thresholds,
        "peak_gpu_allocated_mb": max(
            [initialization_peak_allocated_mb]
            + [row["peak_gpu_allocated_mb"] for row in rows]
        ),
        "peak_gpu_reserved_mb": max(
            [initialization_peak_reserved_mb]
            + [row["peak_gpu_reserved_mb"] for row in rows]
        ),
        "cases": rows,
    }}
    write_terminal(args.terminal, terminal)


if __name__ == "__main__":
    main()
