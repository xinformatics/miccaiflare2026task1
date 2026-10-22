"""Local scorer, byte-identical to the official FLARE Task-1 metric.

Replicates /tmp/flare-eval/FLARE24/T1_FLARE24_DSC_NSD_Eval.py exactly, using the
organisers' vendored SurfaceDice.py. The scoring rules that bite:

  * predictions must be strictly binary -- max(pred) > 1 scores 0/0
  * empty GT + empty pred  -> DSC 1, NSD 1
  * empty GT + any FP      -> DSC 0, NSD 0   (the healthy-case trap)
  * non-empty GT + empty   -> DSC 0, NSD 0
  * DSC < 0.2              -> NSD forced to 0
  * NSD tolerance          = 2 mm

Healthy cases ship without label files, so --healthy scores them against an
implicit all-zero GT: clean -> 1.0, any false positive -> 0.0.

  python code/evaluate.py --pred preds/ --gt validation/Validation-Public-Labels
  python code/evaluate.py --pred preds_healthy/ --healthy
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import nibabel as nb
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import source_of  # noqa: E402
from SurfaceDice import (compute_dice_coefficient,  # noqa: E402
                         compute_surface_dice_at_tolerance,
                         compute_surface_distances)

TUMOR_TOLERANCE = 2
PROJ = Path("/media/shashank/NewVolume/PancancerCTSeg")


def score_case(gt_data, seg_data, spacing):
    """The organisers' exact decision ladder."""
    if np.max(seg_data) > 1:
        return 0.0, 0.0
    gsum, ssum = np.sum(gt_data), np.sum(seg_data)
    if gsum == 0 and ssum == 0:
        return 1.0, 1.0
    if gsum == 0 and ssum > 0:
        return 0.0, 0.0
    if gsum > 0 and ssum == 0:
        return 0.0, 0.0
    dsc = compute_dice_coefficient(gt_data, seg_data)
    if dsc < 0.2:                      # organisers skip NSD when DSC is tiny
        return float(dsc), 0.0
    sd = compute_surface_distances(gt_data, seg_data, spacing)
    return float(dsc), float(compute_surface_dice_at_tolerance(sd, TUMOR_TOLERANCE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="dir of predicted .nii.gz")
    ap.add_argument("--gt", default=str(PROJ / "validation/Validation-Public-Labels"))
    ap.add_argument("--healthy", action="store_true",
                    help="score against implicit all-zero GT (no label files)")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    pred_dir = Path(a.pred)
    rows = []

    if a.healthy:
        names = sorted(p.name for p in pred_dir.glob("*.nii.gz"))
        if not names:
            sys.exit(f"no predictions in {pred_dir}")
        for n in names:
            seg_nii = nb.load(pred_dir / n)
            seg = np.uint8(seg_nii.get_fdata())
            sp = seg_nii.header.get_zooms()[:3]
            gt = np.zeros_like(seg)
            d, s = score_case(gt, seg, sp)
            fp = int((seg > 0).sum())
            rows.append((n[: -len(".nii.gz")], d, s, fp))
            print(f"{n}  DSC {d:.4f}  NSD {s:.4f}  FPvox {fp}", flush=True)
    else:
        gt_dir = Path(a.gt)
        names = sorted(p.name for p in gt_dir.glob("*.nii.gz"))
        if not names:
            sys.exit(f"no ground truth in {gt_dir}")
        for n in names:
            pp = pred_dir / n
            if not pp.exists():
                # a missing prediction is a failed case, exactly as on Codabench
                rows.append((n[: -len(".nii.gz")], 0.0, 0.0, -1))
                print(f"{n}  MISSING PREDICTION -> 0/0", flush=True)
                continue
            gt_nii = nb.load(gt_dir / n)
            sp = gt_nii.header.get_zooms()[:3]
            gt = np.uint8(gt_nii.get_fdata())
            seg = np.uint8(nb.load(pp).get_fdata())
            d, s = score_case(gt, seg, sp)
            rows.append((n[: -len(".nii.gz")], d, s, int((seg > 0).sum())))
            print(f"{n}  DSC {d:.4f}  NSD {s:.4f}", flush=True)

    dsc = np.array([r[1] for r in rows])
    nsd = np.array([r[2] for r in rows])
    tag = "HEALTHY (no-lesion)" if a.healthy else "LESION-BEARING"
    print(f"\n=============== {tag}: n={len(rows)} ===============")
    print(f"DSC  mean {dsc.mean():.4f}  std {dsc.std():.4f}  median {np.median(dsc):.4f}")
    print(f"NSD  mean {nsd.mean():.4f}  std {nsd.std():.4f}  median {np.median(nsd):.4f}")
    if a.healthy:
        clean = int((dsc == 1.0).sum())
        print(f"clean (zero FP): {clean}/{len(rows)}  "
              f"-> false-positive rate {1 - clean / len(rows):.1%}")
    else:
        print(f"cases DSC=0 : {(dsc == 0).sum()}   DSC>0.2: {(dsc > 0.2).sum()}")
        by = defaultdict(list)
        for cid, d, s, _ in rows:
            by[source_of(cid)].append((d, s))
        if len(by) > 1:
            print("\nper-source:")
            for k, v in sorted(by.items()):
                v = np.array(v)
                print(f"  {k:34s} n={len(v):4d}  DSC {v[:,0].mean():.4f}  "
                      f"NSD {v[:,1].mean():.4f}")

    out = Path(a.csv) if a.csv else pred_dir.parent / f"metrics_{pred_dir.name}.csv"
    with open(out, "w") as f:
        f.write("Name,Lesion_DSC,Lesion_NSD,pred_fg_voxels\n")
        for cid, d, s, fp in rows:
            f.write(f"{cid},{round(d,4)},{round(s,4)},{fp}\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
