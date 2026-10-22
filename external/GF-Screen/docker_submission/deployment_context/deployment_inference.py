#!/usr/bin/env python3
"""Run selectable Mayo Clinic deployment ablations from raw challenge CTs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


WORKSPACE = Path("/workspace")


def run(command: list[str], stage: str, rows: list[dict], cwd: str | None = None) -> None:
    started = time.perf_counter()
    print(f"[mayo] stage={stage} start", flush=True)
    subprocess.run(command, check=True, cwd=cwd)
    elapsed = time.perf_counter() - started
    rows.append({"stage": stage, "wall_seconds": elapsed})
    print(f"[mayo] stage={stage} seconds={elapsed:.3f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=WORKSPACE / "inputs")
    parser.add_argument("--output", type=Path, default=WORKSPACE / "outputs")
    parser.add_argument(
        "--mode",
        choices=("gf", "anchor", "proxy", "proxy_stu"),
        default=os.environ.get("MAYO_ABLATION", "gf"),
    )
    args = parser.parse_args()
    focus_batch_size = int(os.environ.get("MAYO_FOCUS_BATCH_SIZE", "1"))
    glance_batch_size = int(os.environ.get("MAYO_GLANCE_BATCH_SIZE", "4"))
    biomed_batch_size = int(os.environ.get("MAYO_BIOMED_BATCH_SIZE", "4"))
    crop_overlap = float(os.environ.get("MAYO_CROP_OVERLAP", "0.75"))
    lesion_threshold = float(os.environ.get("MAYO_LESION_THRESHOLD", "0.60"))
    adaptive_selected_threshold = int(os.environ.get("MAYO_ADAPTIVE_SELECTED_THRESHOLD", "300"))
    if focus_batch_size not in (1, 2, 4, 8):
        raise ValueError("MAYO_FOCUS_BATCH_SIZE must be 1, 2, 4, or 8")
    if glance_batch_size not in (1, 2, 4, 8, 16, 32):
        raise ValueError("MAYO_GLANCE_BATCH_SIZE must be 1, 2, 4, 8, 16, or 32")
    if biomed_batch_size not in (2, 4, 8):
        raise ValueError("MAYO_BIOMED_BATCH_SIZE must be 2, 4, or 8")
    if crop_overlap not in (0.50, 0.625, 0.75):
        raise ValueError("MAYO_CROP_OVERLAP must be 0.50, 0.625, or 0.75")
    if not 0.0 < lesion_threshold < 1.0:
        raise ValueError("MAYO_LESION_THRESHOLD must lie strictly between zero and one")
    if adaptive_selected_threshold < 0:
        raise ValueError("MAYO_ADAPTIVE_SELECTED_THRESHOLD must be nonnegative")
    if not list(args.input.glob("*.nii.gz")):
        raise FileNotFoundError(f"no NIfTI inputs in {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    scratch = WORKSPACE / "runtime"
    if scratch.exists():
        shutil.rmtree(scratch)
    for name in ("biomed_inputs", "whole_npz", "tumor_npz", "whole", "tumor", "bbox", "cbg", "guided", "anchor"):
        (scratch / name).mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    started = time.perf_counter()

    if args.mode == "gf":
        run(
            [
                "python", "-u", "/workspace/external/GF-Screen/docker_submission/predict_dual_glance_shared_focus.py",
                "--images", str(args.input), "--bbox-output", str(scratch / "bbox"),
                "--cbg-output", str(scratch / "cbg"), "--terminal", str(scratch / "gf.json"),
                "--bbox-checkpoint", "/workspace/assets/gf_bbox_full_state.pt",
                "--cbg-checkpoint", "/workspace/assets/gf_cbg_glance_state.pt",
                "--dataset-role", "test", "--gpu", "0", "--workers", "0",
                "--focus-batch-size", str(focus_batch_size),
                "--glance-batch-size", str(glance_batch_size), "--bbox-only",
                "--crop-overlap", str(crop_overlap),
                "--lesion-threshold", str(lesion_threshold),
                "--adaptive-overlap-selected-threshold", str(adaptive_selected_threshold),
            ],
            "gf_bbox",
            rows,
        )
        run(
            [
                "python", "-u", "/workspace/external/GF-Screen/docker_submission/compose_deployment_masks.py",
                "--images", str(args.input), "--bbox", str(scratch / "bbox"),
                "--output", str(args.output), "--mode", "gf",
                "--terminal", str(scratch / "compose.json"),
            ],
            "compose_gf",
            rows,
        )
    else:
        run(
            [
                "python", "-u", "/workspace/external/BiomedParse/miccai_nifti_adapter.py", "prepare",
                "--images", str(args.input), "--output", str(scratch / "biomed_inputs"),
                "--prompt", "whole-body lesion",
            ],
            "biomed_prepare",
            rows,
        )
        run(
            [
                "python", "-u", "/workspace/external/BiomedParse/miccai_dual_prompt_inference.py",
                "--input-dir", str(scratch / "biomed_inputs"),
                "--whole-output-dir", str(scratch / "whole_npz"),
                "--tumor-output-dir", str(scratch / "tumor_npz"),
                "--terminal", str(scratch / "biomed.json"),
                "--checkpoint", "/workspace/assets/biomedparse_3d_state_only.pt",
                "--device", "cuda:0", "--slice-batch-size", str(biomed_batch_size),
                "--existence-threshold", "0.50", "--dataset-role", "test",
            ],
            "biomed_dual_prompt",
            rows,
            cwd="/workspace/external/BiomedParse/container_workspace",
        )
        for route in ("whole", "tumor"):
            run(
                [
                    "python", "-u", "/workspace/external/BiomedParse/miccai_nifti_adapter.py", "convert",
                    "--images", str(args.input), "--predictions", str(scratch / f"{route}_npz"),
                    "--output", str(scratch / route), "--segmentation-key", "segs_e050",
                ],
                f"convert_{route}",
                rows,
            )
        run(
            [
                "python", "-u", "/workspace/external/GF-Screen/docker_submission/predict_dual_glance_shared_focus.py",
                "--images", str(args.input), "--bbox-output", str(scratch / "bbox"),
                "--cbg-output", str(scratch / "cbg"),
                "--guided-proposal", str(scratch / "whole"),
                "--guided-output", str(scratch / "guided"),
                "--terminal", str(scratch / "gf.json"),
                "--bbox-checkpoint", "/workspace/assets/gf_bbox_full_state.pt",
                "--cbg-checkpoint", "/workspace/assets/gf_cbg_glance_state.pt",
                "--dataset-role", "test", "--gpu", "0", "--workers", "0",
                "--focus-batch-size", str(focus_batch_size),
                "--glance-batch-size", str(glance_batch_size),
                "--crop-overlap", str(crop_overlap),
                "--lesion-threshold", str(lesion_threshold),
                "--adaptive-overlap-selected-threshold", str(adaptive_selected_threshold),
            ],
            "gf_shared_focus",
            rows,
        )
        compose_output = args.output if args.mode in ("anchor", "proxy") else scratch / "anchor"
        compose_mode = args.mode if args.mode in ("anchor", "proxy") else "anchor"
        run(
            [
                "python", "-u", "/workspace/external/GF-Screen/docker_submission/compose_deployment_masks.py",
                "--images", str(args.input), "--bbox", str(scratch / "bbox"),
                "--whole", str(scratch / "whole"), "--tumor", str(scratch / "tumor"),
                "--guided", str(scratch / "guided"), "--cbg", str(scratch / "cbg"),
                "--output", str(compose_output), "--mode", compose_mode,
                "--terminal", str(scratch / "compose.json"),
            ],
            f"compose_{compose_mode}",
            rows,
        )
        if args.mode == "proxy_stu":
            run(
                [
                    "python", "-u", "/workspace/external/STU-Net/pancancer/predict_stunet_proxy_sparse.py",
                    "--images", str(args.input), "--anchors", str(scratch / "anchor"),
                    "--bbox-raw", str(scratch / "bbox"), "--prompt-whole", str(scratch / "whole"),
                    "--prompt-tumor", str(scratch / "tumor"), "--output", str(args.output),
                    "--model-dir", "/workspace/nnUNet_results/Dataset501_PanCancer/STUNetTrainerPanCancerComponentBox__nnUNetPlans__3d_fullres_STUNetB_PartialBox", "--checkpoint", "checkpoint.pth",
                    "--gpu", "0", "--part-id", "0", "--num-parts", "1", "--workers", "2",
                    "--dataset-role", "test", "--terminal", str(scratch / "stunet.json"),
                ],
                "stunet_sparse_confirm",
                rows,
            )

    stage_terminals = {}
    for name in ("gf", "biomed", "compose", "stunet"):
        path = scratch / f"{name}.json"
        if path.exists():
            stage_terminals[name] = json.loads(path.read_text(encoding="utf-8"))

    terminal = {
        "status": "COMPLETE",
        "mode": args.mode,
        "focus_batch_size": focus_batch_size,
        "glance_batch_size": glance_batch_size,
        "biomed_batch_size": biomed_batch_size,
        "crop_overlap": crop_overlap,
        "lesion_threshold": lesion_threshold,
        "adaptive_overlap_selected_threshold": adaptive_selected_threshold,
        "case_count": len(list(args.output.glob("*.nii.gz"))),
        "wall_seconds": time.perf_counter() - started,
        "stages": rows,
        "stage_terminals": stage_terminals,
    }
    (args.output / "mayo_runtime.json").write_text(json.dumps(terminal, indent=2) + "\n", encoding="utf-8")
    shutil.rmtree(scratch)
    print(json.dumps(terminal, indent=2), flush=True)


if __name__ == "__main__":
    main()
