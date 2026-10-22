"""Profile the public validation set: geometry, lesion size/HU, anatomical extent.

Drives three decisions: target spacing, HU window for uint8 quantisation, and which
training sources match the scored domain.
"""
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

ROOT = Path("/media/shashank/NewVolume/PancancerCTSeg")
IMG = ROOT / "validation/Validation-Public-Images"
LAB = ROOT / "validation/Validation-Public-Labels"


def body_mask(vol):
    """Rough body mask: air is < -500 HU."""
    m = vol > -500
    m = ndimage.binary_fill_holes(m)
    return m


def main():
    rows = []
    les_hu = []
    for lp in sorted(LAB.glob("*.nii.gz")):
        cid = lp.name[: -len(".nii.gz")]
        ip = IMG / f"{cid}_0000.nii.gz"
        im, lb = nib.load(ip), nib.load(lp)
        vol = np.asanyarray(im.dataobj).astype(np.float32)
        seg = np.asanyarray(lb.dataobj).astype(np.uint8)
        sp = np.array(im.header.get_zooms()[:3], dtype=np.float64)
        voxmm3 = float(np.prod(sp))

        lab, n = ndimage.label(seg > 0)
        nvox = int((seg > 0).sum())
        hu = vol[seg > 0]
        if hu.size:
            les_hu.append(np.percentile(hu, [1, 25, 50, 75, 99]))

        # z-centroid of lesion as a fraction of the scanned body extent
        zfrac = np.nan
        if nvox:
            zc = float(np.argwhere(seg > 0)[:, 2].mean())
            zfrac = zc / max(seg.shape[2] - 1, 1)

        comp_mm3 = sorted(
            (int(c) * voxmm3 for c in np.bincount(lab.ravel())[1:]), reverse=True
        )
        rows.append(
            dict(
                case=cid,
                shape=list(map(int, vol.shape)),
                spacing=[round(float(x), 3) for x in sp],
                fov_mm=[round(float(s * z), 1) for s, z in zip(vol.shape, sp)],
                n_lesions=int(n),
                lesion_mm3=round(nvox * voxmm3, 1),
                largest_mm3=round(comp_mm3[0], 1) if comp_mm3 else 0.0,
                smallest_mm3=round(comp_mm3[-1], 1) if comp_mm3 else 0.0,
                frac_fg=round(nvox / seg.size, 8),
                lesion_z_frac=round(float(zfrac), 3),
                body_hu_p01=round(float(np.percentile(vol[body_mask(vol)], 1)), 1),
                body_hu_p99=round(float(np.percentile(vol[body_mask(vol)], 99)), 1),
            )
        )
        print(f"{cid}  {rows[-1]['shape']}  sp={rows[-1]['spacing']}  "
              f"n={n}  vol={rows[-1]['lesion_mm3']:.0f}mm3", flush=True)

    out = ROOT / "code/val_profile.json"
    out.write_text(json.dumps(rows, indent=1))

    sp = np.array([r["spacing"] for r in rows])
    fov = np.array([r["fov_mm"] for r in rows])
    lv = np.array([r["lesion_mm3"] for r in rows])
    nl = np.array([r["n_lesions"] for r in rows])
    hu = np.array(les_hu)

    def q(a, name, ps=(0, 10, 50, 90, 100)):
        print(f"{name:16s} " + "  ".join(
            f"p{p}={np.percentile(a, p):.2f}" for p in ps))

    print("\n================ PUBLIC VAL PROFILE (n=%d) ================" % len(rows))
    for i, ax in enumerate("xyz"):
        q(sp[:, i], f"spacing {ax} (mm)")
    for i, ax in enumerate("xyz"):
        q(fov[:, i], f"FOV {ax} (mm)")
    q(lv, "lesion vol mm3")
    q(nl.astype(float), "n lesions")
    q(np.array([r["frac_fg"] for r in rows]) * 100, "fg %")
    print("\nlesion HU percentiles (mean over cases):")
    for name, v in zip(["p1", "p25", "p50", "p75", "p99"], hu.mean(0)):
        print(f"  {name}: {v:.1f}")
    print(f"\nlesion z-frac (0=first slice): "
          f"{np.nanmean([r['lesion_z_frac'] for r in rows]):.3f}")
    print(f"cases with 0 lesions: {(nl == 0).sum()}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
