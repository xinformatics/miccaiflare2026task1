"""Package predictions into a Codabench validation-leaderboard submission.

Format taken from the organisers' own baseline submission
(baselines/flare25_baseline_val.zip):
  * flat zip, no directories
  * one file per hidden-validation case, named FLARE-Val_XXXX.nii.gz
    (the input is FLARE-Val_XXXX_0000.nii.gz -- the _0000 channel suffix is dropped)
  * uint8, values in {0, 1}, original geometry preserved

Validated against the scoring rules before zipping, because several of them fail
silently and cost the whole case:
  * max(pred) > 1            -> DSC 0 AND NSD 0   (Information.md / T1 eval code)
  * missing case             -> scored as 0
  * geometry mismatch vs the input image -> unusable

  python code/make_submission.py --pred <dir> --out submission.zip
"""
import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

PROJ = Path("/media/shashank/NewVolume/PancancerCTSeg")
HIDDEN = PROJ / "validation/Validation-Hidden-Images"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="dir of predicted .nii.gz")
    ap.add_argument("--images", default=str(HIDDEN))
    ap.add_argument("--out", default=str(PROJ / "submission.zip"))
    ap.add_argument("--allow-empty", action="store_true",
                    help="package even if every prediction is empty")
    a = ap.parse_args()

    pred_dir, img_dir = Path(a.pred), Path(a.images)
    expected = sorted(p.name.replace("_0000.nii.gz", ".nii.gz")
                      for p in img_dir.glob("*_0000.nii.gz"))
    if not expected:
        sys.exit(f"no *_0000.nii.gz inputs in {img_dir}")

    problems, total_fg, n_empty = [], 0, 0
    print(f"validating {len(expected)} cases from {pred_dir}\n")
    for name in expected:
        p = pred_dir / name
        if not p.is_file():
            problems.append(f"MISSING  {name}")
            continue
        seg = sitk.ReadImage(str(p))
        arr = sitk.GetArrayFromImage(seg)
        img = sitk.ReadImage(str(img_dir / name.replace(".nii.gz", "_0000.nii.gz")))

        if seg.GetSize() != img.GetSize():
            problems.append(f"GEOMETRY {name}: pred {seg.GetSize()} != img {img.GetSize()}")
        u = np.unique(arr)
        if u.max() > 1:
            problems.append(f"NON-BINARY {name}: values {u[:8]} -> would score 0/0")
        if not np.all(np.isin(u, (0, 1))):
            problems.append(f"BAD VALUES {name}: {u[:8]}")
        fg = int((arr > 0).sum())
        total_fg += fg
        n_empty += fg == 0

    print(f"empty predictions : {n_empty}/{len(expected)}")
    print(f"total fg voxels   : {total_fg:,}")
    if problems:
        print(f"\n{len(problems)} PROBLEMS:")
        for x in problems[:20]:
            print("  " + x)
        sys.exit("refusing to package a submission with problems")

    if n_empty == len(expected) and not a.allow_empty:
        sys.exit("every prediction is empty -- this scores 0 on every lesion case. "
                 "Pass --allow-empty if you are deliberately testing the pipeline.")

    out = Path(a.out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in expected:
            z.write(pred_dir / name, arcname=name)   # arcname => flat, no dirs
    mb = out.stat().st_size / 2**20
    print(f"\nwrote {out}  ({mb:.1f} MB, {len(expected)} files, flat)")
    print("upload this to the Codabench validation leaderboard")


if __name__ == "__main__":
    main()
