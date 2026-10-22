"""Mirror of the FP experiment: recover MISSED ground-truth lesions and measure
the DSC vs NSD split, by lesion size.

A wholly missed lesion of radius r costs DSC ~r^3 (volume) but NSD only ~r^2
(surface), so if the champion's deficit is missed detections rather than loose
boundaries, FN recovery should raise DSC MORE than NSD -- the opposite of the
measured FP-removal asymmetry (+0.0789 DSC / +0.0971 NSD). Label-informed
oracle: headroom measurement, not a deployable method.
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
OUT  = ROOT / "external/GF-Screen/analysis/subtractive_gate_diag_20260823"

VAR = {"champion": None,
       "recover ALL missed":        lambda mm3: True,
       "recover missed >=3000mm3":  lambda mm3: mm3 >= 3000,
       "recover missed >=1000mm3":  lambda mm3: mm3 >= 1000,
       "recover missed <1000mm3":   lambda mm3: mm3 < 1000}
res = {k: [] for k in VAR}; rows = []

for p in sorted(PRED.glob("*.nii.gz")):
    gn = nib.load(str(LAB / p.name))
    gt = np.uint8(np.asarray(gn.dataobj) > 0)
    sp = gn.header.get_zooms()[:3]
    pn = nib.load(str(p)); pred = np.asarray(pn.dataobj) > 0
    vox = float(abs(np.linalg.det(pn.affine[:3, :3])))
    glab, ngl = ndimage.label(gt, structure=np.ones((3, 3, 3)))
    missed = []
    for gi in range(1, ngl + 1):
        m = glab == gi; n = int(m.sum())
        if int((m & pred).sum()) < 0.10 * n:      # same 10% detection criterion
            missed.append((gi, n * vox))
    row = {"case": p.name.replace(".nii.gz", "")}
    for vname, rule in VAR.items():
        if rule is None:
            seg = pred
        else:
            add = [gi for gi, mm3 in missed if rule(mm3)]
            seg = pred | np.isin(glab, add) if add else pred
        d, s = score_case(gt, np.uint8(seg), sp)
        res[vname].append((d, s)); row[vname] = f"{d:.4f}/{s:.4f}"
    rows.append(row)
    print(f"{row['case']} missed={len(missed)} " + " ".join(f"{k.split()[0][:6]}={row[k]}" for k in VAR), flush=True)

print("\n" + "=" * 88)
b = np.array(res["champion"])
print(f"{'variant':<28}{'DSC':>9}{'NSD':>9}{'dDSC':>9}{'dNSD':>9}{'dDSC/dNSD':>11}")
for v in VAR:
    a = np.array(res[v]); dd = a[:,0].mean()-b[:,0].mean(); dn = a[:,1].mean()-b[:,1].mean()
    r = f"{dd/dn:.2f}" if abs(dn) > 1e-9 else "-"
    print(f"{v:<28}{a[:,0].mean():>9.4f}{a[:,1].mean():>9.4f}{dd:>+9.4f}{dn:>+9.4f}{r:>11}")
with open(OUT / "fn_recovery_percase.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
