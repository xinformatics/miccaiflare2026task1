"""Per-component subtractive audit of the deployed champion on the fixed 50 public cases.

Read-only, CPU-only, one case in memory at a time. Produces, for every predicted
component: physical volume, TP/FP status at the 10%-overlap detection criterion,
the exact macro-Dice change from deleting that single component, and anatomy /
intensity descriptors that a deployed gate could legally observe.
"""
import csv, json, sys
from pathlib import Path
import numpy as np, nibabel as nib
from scipy import ndimage

ROOT = Path("/mnt/nvme2-data/shashank/MICCAI26")
PRED = ROOT / "external/GF-Screen/public_sweeps/complementary_guided_union_20260821/biomed_new2k_s25"
LAB  = ROOT / "validation/Validation-Public-Labels"
IMG  = ROOT / "validation/Validation-Public-Images"
ANAT = ROOT / "external/GF-Screen/public_sweeps/totalseg_fast_public_20260822"
OUT  = ROOT / "external/GF-Screen/analysis/subtractive_gate_diag_20260823"

def dice(a_sum, b_sum, inter):
    return 0.0 if (a_sum + b_sum) == 0 else 2.0 * inter / (a_sum + b_sum)

rows, case_rows = [], []
cases = sorted(p.name for p in PRED.glob("*.nii.gz"))
for ci, name in enumerate(cases, 1):
    pn = nib.load(str(PRED / name))
    pred = np.asarray(pn.dataobj) > 0
    gt = np.asarray(nib.load(str(LAB / name)).dataobj) > 0
    if pred.shape != gt.shape:
        print(f"SHAPE MISMATCH {name}", file=sys.stderr); continue
    vox_mm3 = float(abs(np.linalg.det(pn.affine[:3, :3])))

    lab, nlab = ndimage.label(pred, structure=np.ones((3, 3, 3)))
    glab, ngl = ndimage.label(gt, structure=np.ones((3, 3, 3)))
    ps, gs = int(pred.sum()), int(gt.sum())
    inter0 = int((pred & gt).sum())
    base = dice(ps, gs, inter0)

    # per-GT-component voxel counts, for the 10% detection criterion
    gt_sizes = ndimage.sum(gt, glab, index=np.arange(1, ngl + 1)).astype(np.int64) if ngl else np.array([])

    anat_p = ANAT / name
    anat = np.asarray(nib.load(str(anat_p).replace('.nii.gz', '.nii.gz')).dataobj).astype(np.int16) if anat_p.exists() else None
    imgp = IMG / name.replace('.nii.gz', '_0000.nii.gz')
    img = np.asarray(nib.load(str(imgp)).dataobj).astype(np.float32) if imgp.exists() else None

    kept_fp = 0
    for k in range(1, nlab + 1):
        m = lab == k
        n = int(m.sum())
        i = int((m & gt).sum())
        # which GT components does it cover >=10% of?
        covered = 0
        if i:
            for gi in np.unique(glab[m]):
                if gi == 0: continue
                ov = int((m & (glab == gi)).sum())
                if ov >= 0.10 * gt_sizes[gi - 1]: covered += 1
        is_tp = i > 0
        d_removed = dice(ps - n, gs, inter0 - i)
        r = dict(case=name.replace('.nii.gz', ''), comp=k, voxels=n,
                 mm3=n * vox_mm3, overlap_voxels=i,
                 overlap_frac=(i / n if n else 0.0), gt_comps_detected=covered,
                 is_tp=int(is_tp), dice_if_removed=d_removed,
                 delta_dice_if_removed=d_removed - base)
        if anat is not None and anat.shape == pred.shape:
            av = anat[m]
            nz = av[av > 0]
            r["anat_bg_frac"] = float((av == 0).mean())
            if nz.size:
                u, c = np.unique(nz, return_counts=True)
                r["anat_top_label"] = int(u[c.argmax()]); r["anat_top_frac"] = float(c.max() / av.size)
            else:
                r["anat_top_label"] = 0; r["anat_top_frac"] = 0.0
        if img is not None and img.shape == pred.shape:
            iv = img[m]
            r["hu_mean"] = float(iv.mean()); r["hu_std"] = float(iv.std())
            r["hu_p10"] = float(np.percentile(iv, 10)); r["hu_p90"] = float(np.percentile(iv, 90))
        rows.append(r)
        if not is_tp: kept_fp += 1

    case_rows.append(dict(case=name.replace('.nii.gz', ''), base_dice=base,
                          pred_comps=nlab, gt_comps=ngl, fp_comps=kept_fp,
                          fp_free_dice=dice(inter0, gs, inter0)))
    print(f"[{ci}/50] {name} dice={base:.4f} comps={nlab} fp={kept_fp}", flush=True)

OUT.mkdir(parents=True, exist_ok=True)
keys = sorted({k for r in rows for k in r})
with open(OUT / "components.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
with open(OUT / "cases.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=sorted(case_rows[0])); w.writeheader(); w.writerows(case_rows)
print(json.dumps(dict(components=len(rows), cases=len(case_rows),
                      macro_base=float(np.mean([c["base_dice"] for c in case_rows])),
                      macro_fp_free=float(np.mean([c["fp_free_dice"] for c in case_rows]))), indent=1))
