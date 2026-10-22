"""GT-side appearance audit: are MISSED lesions distinguished by low CT contrast?

For every GT lesion component on the fixed 50 public cases, record physical
volume, whether the deployed champion detected it (>=10% overlap), its interior
HU statistics, and the HU contrast against a local shell of surrounding tissue.
Read-only, CPU-only, one case at a time.
"""
import csv, json
from pathlib import Path
import numpy as np, nibabel as nib
from scipy import ndimage

ROOT = Path("/mnt/nvme2-data/shashank/MICCAI26")
PRED = ROOT / "external/GF-Screen/public_sweeps/complementary_guided_union_20260821/biomed_new2k_s25"
LAB  = ROOT / "validation/Validation-Public-Labels"
IMG  = ROOT / "validation/Validation-Public-Images"
OUT  = ROOT / "external/GF-Screen/analysis/subtractive_gate_diag_20260823"

rows = []
for ci, p in enumerate(sorted(PRED.glob("*.nii.gz")), 1):
    name = p.name
    pred = np.asarray(nib.load(str(p)).dataobj) > 0
    ln = nib.load(str(LAB / name))
    gt = np.asarray(ln.dataobj) > 0
    img = np.asarray(nib.load(str(IMG / name.replace('.nii.gz', '_0000.nii.gz'))).dataobj).astype(np.float32)
    vox = float(abs(np.linalg.det(ln.affine[:3, :3])))
    glab, ngl = ndimage.label(gt, structure=np.ones((3, 3, 3)))
    for gi in range(1, ngl + 1):
        m = glab == gi
        n = int(m.sum())
        ov = int((m & pred).sum())
        # local shell: dilate the component box and take non-lesion tissue around it
        sl = ndimage.find_objects(m.astype(np.uint8))[0]
        pad = 6
        ex = tuple(slice(max(0, s.start - pad), min(d, s.stop + pad))
                   for s, d in zip(sl, m.shape))
        sub_m, sub_i, sub_gt = m[ex], img[ex], gt[ex]
        shell = ndimage.binary_dilation(sub_m, iterations=3) & ~sub_gt
        iv = img[m]
        sv = sub_i[shell] if shell.any() else np.array([np.nan])
        rows.append(dict(
            case=name.replace('.nii.gz', ''), gt_comp=gi, mm3=n * vox,
            detected=int(ov >= 0.10 * n), overlap_frac=ov / n,
            hu_mean=float(iv.mean()), hu_median=float(np.median(iv)),
            hu_std=float(iv.std()),
            shell_hu_median=float(np.nanmedian(sv)),
            contrast=float(np.median(iv) - np.nanmedian(sv)),
            abs_contrast=float(abs(np.median(iv) - np.nanmedian(sv))),
        ))
    print(f"[{ci}/50] {name} gt_comps={ngl}", flush=True)

with open(OUT / "gt_components.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(json.dumps(dict(gt_components=len(rows),
                      detected=sum(r["detected"] for r in rows)), indent=1))
