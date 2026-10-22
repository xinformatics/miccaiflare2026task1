"""Price the additive selector: raw components the deployed champion discarded.

Vectorised - a fixed number of full-volume passes per case instead of one per
component. For each novel raw component: new voxels contributed, whether it hits
a lesion, size of the largest lesion hit, and the exact macro-DSC change from
adding it alone. Yields gain-per-correct vs loss-per-wrong and the minimum
precision an additive selector needs - the mirror of the deletion bar.
"""
import csv
from pathlib import Path
import numpy as np, nibabel as nib
from scipy import ndimage

ROOT = Path("/mnt/nvme2-data/shashank/MICCAI26")
GF = ROOT / "external/GF-Screen"
DEP = GF / "public_sweeps/complementary_guided_union_20260821/biomed_new2k_s25"
RAW = GF / "predictions/gf_a055_bbox_pu_best_ep280/public_raw"
LAB = ROOT / "validation/Validation-Public-Labels"
OUT = GF / "analysis/subtractive_gate_diag_20260823"

rows = []
for p in sorted(DEP.glob("*.nii.gz")):
    rp = RAW / p.name
    if not rp.exists():
        continue
    dn = nib.load(str(p))
    dep = np.asarray(dn.dataobj) > 0
    raw = np.asarray(nib.load(str(rp)).dataobj) > 0
    gt = np.asarray(nib.load(str(LAB / p.name)).dataobj) > 0
    if raw.shape != dep.shape or gt.shape != dep.shape:
        continue
    vox = float(abs(np.linalg.det(dn.affine[:3, :3])))
    ps, gs = int(dep.sum()), int(gt.sum())
    inter0 = int((dep & gt).sum())
    base = 0.0 if ps + gs == 0 else 2 * inter0 / (ps + gs)

    lab, nl = ndimage.label(raw, structure=np.ones((3, 3, 3)))
    if nl == 0:
        continue
    idx = np.arange(1, nl + 1)
    novel = raw & ~dep
    new_n = ndimage.sum(novel, lab, index=idx)
    new_gt = ndimage.sum(novel & gt, lab, index=idx)

    glab, ngl = ndimage.label(gt, structure=np.ones((3, 3, 3)))
    if ngl:
        gsz = ndimage.sum(gt, glab, index=np.arange(1, ngl + 1)) * vox
        szmap = np.zeros(glab.shape, np.float32)
        nzm = glab > 0
        szmap[nzm] = gsz[glab[nzm] - 1]
        big = ndimage.maximum(szmap, lab, index=idx)
    else:
        big = np.zeros(nl)

    for j in range(nl):
        n = int(new_n[j])
        if n == 0:
            continue
        a = int(new_gt[j])
        d_new = 2 * (inter0 + a) / ((ps + n) + gs)
        rows.append(dict(case=p.name.replace(".nii.gz", ""), comp=j + 1,
                         new_mm3=n * vox, added_gt_voxels=a, hits_gt=int(a > 0),
                         largest_gt_hit_mm3=float(big[j]) if a > 0 else 0.0,
                         delta_dice=d_new - base))
    print(f"{p.name} raw={nl} novel={int((new_n > 0).sum())}", flush=True)

with open(OUT / "raw_addition_candidates.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

N = 50
T_all = [r for r in rows if r["hits_gt"]]
print("\n" + "=" * 92)
print(f"novel raw candidates: {len(rows)}   hitting a lesion: {len(T_all)} ({len(T_all)/len(rows):.1%})")
print(f"\n{'candidate class':<32}{'n':>6}{'nTrue':>7}{'prec':>7}{'gain/true':>11}{'loss/false':>11}{'minPrec':>9}")
for nm, f_ in [("all novel", lambda r: True),
               ("novel >=10000 mm3", lambda r: r["new_mm3"] >= 10000),
               ("novel >=3000 mm3", lambda r: r["new_mm3"] >= 3000),
               ("novel >=1000 mm3", lambda r: r["new_mm3"] >= 1000),
               ("novel <1000 mm3", lambda r: r["new_mm3"] < 1000)]:
    S = [r for r in rows if f_(r)]
    T = [r for r in S if r["hits_gt"]]
    F = [r for r in S if not r["hits_gt"]]
    if not T or not F:
        print(f"{nm:<32}{len(S):>6}{len(T):>7}   (degenerate)")
        continue
    g = np.mean([r["delta_dice"] for r in T]) / N
    l = -np.mean([r["delta_dice"] for r in F]) / N
    rt = l / g if g > 0 else float("inf")
    print(f"{nm:<32}{len(S):>6}{len(T):>7}{len(T)/len(S):>7.1%}{g:>+11.6f}{-l:>+11.6f}{rt/(1+rt):>8.1%}")
print(f"\noracle: add every true novel candidate -> +{sum(r['delta_dice'] for r in T_all)/N:.4f} macro DSC")
