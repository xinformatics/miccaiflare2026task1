"""Does GROWING the champion's already-detected components recover DSC while
holding NSD? Tests the under-segmentation hypothesis implied by
micro precision 0.847 > recall 0.788 and 0.792 voxel recall on >=3000mm3 lesions.

Variants: plain morphological dilation, and intensity-gated dilation that only
expands into voxels whose HU lies inside the component's own [p5,p95] range --
a label-free proxy for lowering the Focus threshold locally. Existing voxels are
never removed. Scored with the official FLARE metric.
"""
import sys, csv
from pathlib import Path
import numpy as np, nibabel as nib
from scipy import ndimage

ROOT = Path("/mnt/nvme2-data/shashank/MICCAI26")
sys.path.insert(0, str(ROOT / "handoff_bundle/code"))
from evaluate import score_case

PRED = ROOT / "external/GF-Screen/public_sweeps/complementary_guided_union_20260821/biomed_new2k_s25"
LAB  = ROOT / "validation/Validation-Public-Labels"
IMG  = ROOT / "validation/Validation-Public-Images"
OUT  = ROOT / "external/GF-Screen/analysis/subtractive_gate_diag_20260823"

VARIANTS = ["champion", "dilate1", "dilate2", "dilate3", "hu_gated2", "hu_gated4"]
res = {v: [] for v in VARIANTS}
rows = []

for p in sorted(PRED.glob("*.nii.gz")):
    gn = nib.load(str(LAB / p.name))
    gt = np.uint8(np.asarray(gn.dataobj) > 0)
    sp = gn.header.get_zooms()[:3]
    pn = nib.load(str(p))
    base = np.asarray(pn.dataobj) > 0
    img = np.asarray(nib.load(str(IMG / p.name.replace('.nii.gz', '_0000.nii.gz'))).dataobj).astype(np.float32)
    st = np.ones((3, 3, 3), bool)

    out = {"champion": base}
    for k in (1, 2, 3):
        out[f"dilate{k}"] = ndimage.binary_dilation(base, st, iterations=k)
    # intensity-gated: allow growth only into voxels inside the predicted
    # foreground's own HU interval (label-free)
    if base.any():
        fv = img[base]
        lo, hi = np.percentile(fv, 5), np.percentile(fv, 95)
        ok = (img >= lo) & (img <= hi)
        for k in (2, 4):
            out[f"hu_gated{k}"] = ndimage.binary_dilation(base, st, iterations=k, mask=ok) | base
    else:
        for k in (2, 4): out[f"hu_gated{k}"] = base

    row = {"case": p.name.replace(".nii.gz", "")}
    for v in VARIANTS:
        d, s = score_case(gt, np.uint8(out[v]), sp)
        res[v].append((d, s))
        row[v] = f"{d:.4f}/{s:.4f}"
    rows.append(row)
    print(f"{row['case']}  " + "  ".join(f"{v}={row[v]}" for v in VARIANTS), flush=True)

print("\n" + "=" * 78)
b = np.array(res["champion"])
print(f"{'variant':<14}{'DSC':>9}{'NSD':>9}{'dDSC':>9}{'dNSD':>9}{'DSC<0.2':>9}")
for v in VARIANTS:
    a = np.array(res[v])
    print(f"{v:<14}{a[:,0].mean():>9.4f}{a[:,1].mean():>9.4f}"
          f"{a[:,0].mean()-b[:,0].mean():>+9.4f}{a[:,1].mean()-b[:,1].mean():>+9.4f}{(a[:,0]<0.2).sum():>9d}")
with open(OUT / "boundary_growth_percase.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
