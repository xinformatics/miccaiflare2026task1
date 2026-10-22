"""Inference + postprocessing, built for the challenge's efficiency budget.

Deliberate choices, each tied to a scoring rule in Information.md:
  * mirroring TTA OFF by default  -- 8x the forward passes for a small DSC gain;
    the 60s/case cliff and the GPU-memory-time integral both punish it.
  * fp16 autocast, single fold    -- a 5-fold ensemble cannot hold 4GB.
  * output written as uint8 0/1   -- max(pred) > 1 scores 0/0.
  * connected-component filtering -- drops specks below --min-mm3. On a healthy case
    a single false-positive voxel costs the whole case (DSC 0 AND NSD 0), so this is
    worth real DSC on the lesion-bearing set.

  python code/predict.py -i imgs/ -o preds/ -d 501 -c 3d_fullres_S4D2W32 \
      -tr nnUNetTrainer_Epoch2000_Lr1e3 -f all --min-mm3 500
"""
import argparse
import os

# torch.compile must be OFF for inference. env.sh exports nnUNet_compile=1 for the
# 1.24x training speedup, but the inductor path resolves CUDA_HOME to the anaconda
# prefix (which has no CUDA runtime) and aborts with a heap corruption:
#   "No CUDA runtime is found, using CUDA_HOME='/home/.../anaconda3'"
#   "malloc(): unaligned tcache chunk detected"  -> SIGABRT
# Compiling buys nothing here anyway: each case is a fresh set of shapes.
os.environ["nnUNet_compile"] = "0"

import time
from pathlib import Path

import numpy as np
import torch


def postprocess(seg, spacing_mm, min_mm3):
    """Strictly binary output; drop connected components below min_mm3.

    `spacing_mm` is only ever consumed via np.prod (voxel volume), so axis order
    does not matter here. It DOES matter if you add an anisotropic rule later:
    `seg` is numpy (z, y, x) while SimpleITK's GetSpacing() is (x, y, z).
    """
    seg = (np.asarray(seg) > 0).astype(np.uint8)
    if min_mm3 <= 0 or seg.max() == 0:
        return seg
    from scipy import ndimage
    lab, n = ndimage.label(seg)
    if n == 0:
        return seg
    vox_mm3 = float(np.prod(np.abs(spacing_mm)))
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    keep = np.zeros(counts.shape[0], dtype=bool)
    keep[counts * vox_mm3 >= min_mm3] = True
    return keep[lab].astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-d", "--dataset", default="501")
    ap.add_argument("-c", "--config", default="3d_fullres_S4D2W32")
    ap.add_argument("-tr", "--trainer", default="nnUNetTrainer_Epoch2000_Lr1e3")
    ap.add_argument("-p", "--plans", default="nnUNetPlans")
    ap.add_argument("-f", "--folds", default="all")
    ap.add_argument("--checkpoint", default="checkpoint_final.pth")
    ap.add_argument("--min-mm3", type=float, default=0.0,
                    help="drop connected components smaller than this (mm^3)")
    ap.add_argument("--mirroring", action="store_true", help="enable TTA (slow)")
    ap.add_argument("--step", type=float, default=0.5, help="tile step size")
    ap.add_argument("--cpu-export", action="store_true",
                    help="resample logits back to original resolution on CPU. This "
                         "is the main GPU-memory lever: nnU-Net otherwise holds a "
                         "full-resolution float logit map on device, which is what "
                         "pushes published solutions from ~2GB to ~7GB peak.")
    ap.add_argument("--model-dir", default=None,
                    help="explicit trained-model dir (overrides -d/-tr/-p/-c)")
    a = ap.parse_args()

    import SimpleITK as sitk
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.file_path_utilities import get_output_folder

    in_dir, out_dir = Path(a.input), Path(a.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = a.model_dir or get_output_folder(a.dataset, a.trainer, a.plans, a.config)
    folds = (a.folds,) if a.folds == "all" else tuple(int(x) for x in a.folds.split(","))

    predictor = nnUNetPredictor(
        tile_step_size=a.step,
        use_gaussian=True,
        use_mirroring=a.mirroring,
        perform_everything_on_device=not a.cpu_export,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(model_dir, folds,
                                                   checkpoint_name=a.checkpoint)
    print(f"model: {model_dir}  folds={folds}  mirroring={a.mirroring}")

    cases = sorted(p for p in in_dir.iterdir() if p.name.endswith(".nii.gz"))
    print(f"{len(cases)} cases -> {out_dir}\n")

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    times = []
    for p in cases:
        name = p.name.replace("_0000.nii.gz", ".nii.gz")
        t0 = time.perf_counter()
        img = sitk.ReadImage(str(p))
        # sequential: no worker pools spawned per case, so the timing is honest
        seg = predictor.predict_from_files_sequential(
            [[str(p)]], None, save_probabilities=False, overwrite=True,
            folder_with_segs_from_prev_stage=None,
        )[0]
        # GetSpacing() is (x,y,z); seg is (z,y,x). Reverse so the axes line up.
        seg = postprocess(seg, img.GetSpacing()[::-1], a.min_mm3)
        out = sitk.GetImageFromArray(seg.astype(np.uint8))
        out.CopyInformation(img)
        sitk.WriteImage(out, str(out_dir / name), useCompression=True)
        dt = time.perf_counter() - t0
        times.append(dt)
        fg = int(seg.sum())
        flag = "!!" if dt > 60 else "  "
        print(f"{flag}{name}  {dt:6.2f}s  fg={fg}", flush=True)

    t = np.array(times)
    print(f"\ntime/case: mean {t.mean():.2f}s  max {t.max():.2f}s  "
          f"over-60s: {(t > 60).sum()}")
    if torch.cuda.is_available():
        print(f"peak VRAM: {torch.cuda.max_memory_allocated()/2**20:.0f} MB "
              f"(budget 4096 MB)")


if __name__ == "__main__":
    main()
