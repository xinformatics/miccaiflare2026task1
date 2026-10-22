#!/usr/bin/env python3
"""Controlled BiomedParse v2 inference for MICCAI NIfTI-adapter NPZs."""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OFFICIAL = ROOT / "container_workspace"
# This repository also contains the newer GitHub `src` tree. The released
# challenge checkpoint/config targets the exact `src` bundled in its image,
# which must take precedence regardless of how this adapter is invoked.
sys.path.insert(0, str(OFFICIAL))

import hydra
import numpy as np
import torch
import torch.nn.functional as functional
from hydra import compose
from hydra.core.global_hydra import GlobalHydra

from inference import merge_multiclass_masks, postprocess
from utils import process_input, process_output


CHECKPOINT = OFFICIAL / "model_weights" / "biomedparse_3D_AllData_MultiView_edge.ckpt"


def load_model(
    device: torch.device, checkpoint: Path | None = None
) -> torch.nn.Module:
    """Instantiate the released model and load one explicit inference asset.

    ``checkpoint`` is optional for backwards compatibility with every frozen
    historical command.  Deployment callers pass the hash-verified compact
    state-only checkpoint, whose ``state_dict`` is tensor-identical to the
    released Lightning checkpoint.
    """
    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(
        config_dir=str(OFFICIAL / "configs"), job_name="miccai_prediction", version_base=None
    ):
        config = compose(config_name="seem_model_3D")
    model = hydra.utils.instantiate(config, _convert_="object")
    model.load_pretrained(str(CHECKPOINT if checkpoint is None else checkpoint))
    return model.to(device).eval()


def predict_case(
    model: torch.nn.Module,
    image: np.ndarray,
    prompts: list[str],
    device: torch.device,
    slice_batch_size: int,
    existence_thresholds: list[float],
    do_nms: bool,
) -> tuple[dict[float, np.ndarray], np.ndarray, float]:
    prepared, pad_width, padded_size, valid_axis = process_input(image, 512)
    input_tensor = {
        "image": prepared.to(device).int().unsqueeze(0),
        "text": ["[SEP]".join(prompts)],
    }
    start = time.perf_counter()
    with torch.inference_mode():
        output = model(input_tensor, mode="eval", slice_batch_size=slice_batch_size)
        logits = output["predictions"]["pred_gmasks"]
        logits = functional.interpolate(
            logits, size=(512, 512), mode="bicubic", align_corners=False, antialias=True
        )
        segmentations = {}
        for threshold in existence_thresholds:
            masks = postprocess(
                logits,
                output["predictions"]["object_existence"],
                threshold=threshold,
                do_nms=do_nms,
            )
            segmentation = merge_multiclass_masks(
                masks, list(range(1, len(prompts) + 1))
            )
            segmentation = process_output(
                segmentation, pad_width, padded_size, valid_axis
            )
            segmentations[threshold] = (segmentation > 0).astype(np.uint8)
        existence = output["predictions"]["object_existence"].sigmoid().detach().cpu().numpy()
    elapsed = time.perf_counter() - start
    return segmentations, existence.astype(np.float16), elapsed


def threshold_key(threshold: float) -> str:
    """Stable NPZ field name for a slice-existence operating point."""
    return f"segs_e{int(round(threshold * 100)):03d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--slice-batch-size", type=int, default=4)
    parser.add_argument(
        "--existence-threshold",
        type=float,
        action="append",
        dest="existence_thresholds",
        help="repeat to emit several slice-existence operating points in one pass",
    )
    parser.add_argument("--disable-nms", action="store_true")
    parser.add_argument("--continue-prediction", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    args = parser.parse_args()
    if args.slice_batch_size < 1:
        parser.error("--slice-batch-size must be positive")
    thresholds = args.existence_thresholds or [0.5]
    thresholds = list(dict.fromkeys(thresholds))
    if any(not 0 <= threshold <= 1 for threshold in thresholds):
        parser.error("--existence-threshold must be in [0,1]")
    keys = [threshold_key(threshold) for threshold in thresholds]
    if len(set(keys)) != len(keys):
        parser.error("existence thresholds must differ by at least 0.01")

    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        parser.error(f"invalid shard {args.shard_id}/{args.num_shards}")
    files = sorted(args.input_dir.glob("*.npz"))[args.shard_id :: args.num_shards]
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"no NPZ files in {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = load_model(device)
    print(f"model ready on {device}; cases={len(files)} prompts={args.prompt}", flush=True)

    for index, source in enumerate(files, 1):
        destination = args.output_dir / source.name
        if args.continue_prediction and destination.exists():
            try:
                with np.load(destination, allow_pickle=False) as existing:
                    complete = all(key in existing for key in keys)
                    complete &= "object_existence" in existing
                    complete &= "existence_thresholds" in existing
                if complete:
                    print(
                        f"skipped {index}/{len(files)} {source.name}: complete",
                        flush=True,
                    )
                    continue
            except (OSError, ValueError):
                pass
            print(
                f"recomputing {index}/{len(files)} {source.name}: incomplete output",
                flush=True,
            )
        with np.load(source, allow_pickle=False) as archive:
            image = np.asarray(archive["imgs"])
        segmentations, existence, elapsed = predict_case(
            model,
            image,
            args.prompt,
            device,
            args.slice_batch_size,
            thresholds,
            not args.disable_nms,
        )
        fields = {
            threshold_key(threshold): segmentations[threshold]
            for threshold in thresholds
        }
        # Preserve compatibility with the official converter and older audit
        # tooling: `segs` aliases the first requested operating point.
        fields["segs"] = segmentations[thresholds[0]]
        fields.update(
            object_existence=existence,
            existence_thresholds=np.asarray(thresholds, dtype=np.float32),
            inference_seconds=np.asarray(elapsed, dtype=np.float32),
            prompts=np.asarray(args.prompt),
        )
        temporary = destination.with_name(f".{destination.name}.partial.npz")
        np.savez_compressed(temporary, **fields)
        temporary.replace(destination)
        print(
            f"predicted {index}/{len(files)} {source.name}: "
            f"foreground={{{', '.join(f'{threshold:.2f}: {int(segmentations[threshold].sum())}' for threshold in thresholds)}}} "
            f"time={elapsed:.2f}s",
            flush=True,
        )
        del image, segmentations, existence
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
