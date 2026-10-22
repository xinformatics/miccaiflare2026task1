"""Zero-shot VISTA3D baseline on the public validation set.

VISTA3D (MONAI, CVPR 2025) is a 3D CT segmentation foundation model trained on
11,454 volumes / 127 classes. It does automatic class-prompted segmentation, so
unlike SAM-style models it needs no human clicks -- which makes it the only fair
"foundation model, no training" reference for this task.

Coverage caveat, stated up front: VISTA3D has only 5 lesion classes --
    23 lung tumor, 24 pancreatic tumor, 26 hepatic tumor,
    27 colon cancer primaries, 128 bone lesion
There is no kidney, adrenal, oesophageal, endometrial or lymph-node tumor class.
Our validation set is whole-body pan-cancer, so VISTA3D structurally cannot score
well on the cases whose lesion type it was never given a prompt for. Read the number
as a floor for "generic foundation model, zero shot", not as a ceiling for the task.

The prompted classes are unioned into one binary lesion mask, then scored with the
organisers' metric via code/evaluate.py.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJ = Path("/media/shashank/NewVolume/PancancerCTSeg")
LESION_CLASSES = [23, 24, 26, 27, 128]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=os.environ.get("HDD", "/mnt/yadav/PancancerCTSeg")
                    + "/bundles/vista3d")
    ap.add_argument("--images", default=str(PROJ / "validation/Validation-Public-Images"))
    ap.add_argument("--raw-out", default=None, help="VISTA3D multi-class output dir")
    ap.add_argument("--out", default=None, help="binary mask output dir")
    ap.add_argument("--classes", default=",".join(map(str, LESION_CLASSES)))
    ap.add_argument("--skip-infer", action="store_true",
                    help="reuse an existing raw output dir, only rebuild binary masks")
    a = ap.parse_args()

    hdd = Path(os.environ.get("HDD", "/mnt/yadav/PancancerCTSeg"))
    raw_out = Path(a.raw_out) if a.raw_out else hdd / "preds/vista3d_raw"
    out = Path(a.out) if a.out else hdd / "preds/vista3d_public"
    classes = [int(x) for x in a.classes.split(",")]
    raw_out.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    if not a.skip_infer:
        cmd = [
            sys.executable, "-m", "monai.bundle", "run",
            "--config_file", f"['{a.bundle}/configs/inference.json',"
                             f"'{a.bundle}/configs/batch_inference.json']",
            "--bundle_root", a.bundle,
            "--input_dir", a.images,
            "--output_dir", str(raw_out),
            "--everything_labels", str(classes),
        ]
        print("prompting VISTA3D with classes:", classes)
        print(" ".join(cmd), flush=True)
        t0 = time.perf_counter()
        p = subprocess.run(cmd, cwd=a.bundle)
        print(f"\ninference wall time {time.perf_counter()-t0:.0f}s (rc={p.returncode})")
        if p.returncode != 0:
            sys.exit("VISTA3D inference failed")

    # --- collapse the multi-class output to a binary lesion mask ---
    import SimpleITK as sitk
    produced = sorted(raw_out.rglob("*.nii.gz"))
    print(f"\nfound {len(produced)} VISTA3D outputs -> binarising into {out}")
    n_empty = 0
    for p in produced:
        # bundle writes <case>/<case>_trans.nii.gz; recover the FLARE case id
        cid = p.name.replace("_trans.nii.gz", "").replace(".nii.gz", "")
        cid = cid.replace("_0000", "")
        img = sitk.ReadImage(str(p))
        arr = sitk.GetArrayFromImage(img)
        mask = np.isin(arr, classes).astype(np.uint8)
        if mask.sum() == 0:
            n_empty += 1
        o = sitk.GetImageFromArray(mask)
        o.CopyInformation(img)
        sitk.WriteImage(o, str(out / f"{cid}.nii.gz"), useCompression=True)
        print(f"  {cid}: fg={int(mask.sum())}", flush=True)
    print(f"\n{n_empty}/{len(produced)} predicted completely empty")
    print(f"\nnow score with:\n  $PY code/evaluate.py --pred {out}")


if __name__ == "__main__":
    main()
