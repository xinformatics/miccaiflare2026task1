"""Score FP-deletion variants of the champion on the fixed 50 public cases with
the official FLARE metric, split by FP component size.

Tests whether large false positives damage DSC disproportionately to NSD, and
how much NSD is unlocked by cases crossing the DSC<0.2 cliff. The deletion rule
is label-informed (an oracle), so these are headroom measurements, not scores.
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

VARIANTS = {
    "champion (baseline)":            None,
    "delete ALL pure-FP":             lambda mm3: True,
    "delete pure-FP >= 20000 mm3":    lambda mm3: mm3 >= 20000,
    "delete pure-FP >= 5000 mm3":     lambda mm3: mm3 >= 5000,
    "delete pure-FP <  5000 mm3":     lambda mm3: mm3 < 5000,
}
res = {k: [] for k in VARIANTS}
percase = []

for p in sorted(PRED.glob("*.nii.gz")):
    gn = nib.load(str(LAB / p.name))
    gt = np.uint8(np.asarray(gn.dataobj) > 0)
    sp = gn.header.get_zooms()[:3]
    pn = nib.load(str(p))
    pred = np.uint8(np.asarray(pn.dataobj) > 0)
    vox = float(abs(np.linalg.det(pn.affine[:3, :3])))
    lab, nl = ndimage.label(pred, structure=np.ones((3, 3, 3)))
    comps = []
    for k in range(1, nl + 1):
        m = lab == k
        comps.append((k, int(m.sum()), int((m & (gt > 0)).sum())))
    row = {"case": p.name.replace(".nii.gz", "")}
    for vname, rule in VARIANTS.items():
        if rule is None:
            seg = pred
        else:
            drop = [k for k, n, i in comps if i == 0 and rule(n * vox)]
            seg = pred.copy()
            if drop: seg[np.isin(lab, drop)] = 0
        d, s = score_case(gt, seg, sp)
        res[vname].append((d, s))
        row[vname] = f"{d:.4f}/{s:.4f}"
    percase.append(row)
    print(f"{row['case']}  " + "  ".join(f"{k.split('(')[0][:14]}={row[k]}" for k in VARIANTS), flush=True)

print("\n" + "=" * 92)
print(f"{'variant':<30}{'DSC':>9}{'NSD':>9}{'dDSC':>9}{'dNSD':>9}{'DSC<0.2':>9}{'NSD=0':>8}")
b = np.array(res["champion (baseline)"])
for vname in VARIANTS:
    a = np.array(res[vname])
    print(f"{vname:<30}{a[:,0].mean():>9.4f}{a[:,1].mean():>9.4f}"
          f"{a[:,0].mean()-b[:,0].mean():>+9.4f}{a[:,1].mean()-b[:,1].mean():>+9.4f}"
          f"{(a[:,0]<0.2).sum():>9d}{(a[:,1]==0).sum():>8d}")
with open("analysis/subtractive_gate_diag_20260823/fp_deletion_percase.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(percase[0])); w.writeheader(); w.writerows(percase)
