"""Find images whose direction cosines are not orthonormal.

ITK (and therefore SimpleITK, and therefore nnU-Net) hard-refuses these:
    "ITK only supports orthonormal direction cosines. No orthonormal definition found!"
nibabel loads them happily, which is why the first integrity pass missed them and
nnUNetv2_extract_fingerprint died at ~92%.

Header-only via nibabel, so this is fast. affine[:3,:3] == R @ diag(spacing); the
direction matrix R is orthonormal iff R.T @ R == I.
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

PROJ = Path("/media/shashank/NewVolume/PancancerCTSeg")
DST = PROJ / "nnUNet_raw/Dataset501_PanCancer"
TOL = 1e-6          # ITK's own check is tight; report the actual deviation


def deviation(path):
    """Max |R^T R - I| for the direction matrix of this file."""
    im = nib.load(str(path))
    A = np.asarray(im.affine[:3, :3], dtype=np.float64)
    sp = np.linalg.norm(A, axis=0)
    if not np.all(np.isfinite(sp)) or np.any(sp <= 0):
        return float("inf")
    R = A / sp
    return float(np.abs(R.T @ R - np.eye(3)).max())


def check(cid):
    out = {"case": cid}
    for tag, p in (("image", DST / f"imagesTr/{cid}_0000.nii.gz"),
                   ("label", DST / f"labelsTr/{cid}.nii.gz")):
        try:
            out[tag] = deviation(p)
        except Exception as e:
            out[tag] = f"ERR:{type(e).__name__}"
    return out


def main():
    cids = sorted(p.name[: -len(".nii.gz")] for p in (DST / "labelsTr").iterdir()
                  if p.name.endswith(".nii.gz"))
    print(f"checking direction cosines for {len(cids)} cases ...", flush=True)

    res = []
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 2)) as ex:
        for i, r in enumerate(ex.map(check, cids, chunksize=64), 1):
            res.append(r)
            if i % 4000 == 0:
                print(f"  {i}/{len(cids)}", flush=True)

    def bad(v):
        return isinstance(v, str) or not np.isfinite(v) or v > TOL

    flagged = [r for r in res if bad(r["image"]) or bad(r["label"])]
    (PROJ / "code/nonorthonormal.json").write_text(
        json.dumps(sorted(r["case"] for r in flagged), indent=1))

    print(f"\n=========== NON-ORTHONORMAL: {len(flagged)} / {len(res)} ===========")
    from collections import Counter
    import sys
    sys.path.insert(0, str(PROJ / "code"))
    from common import source_of
    for k, v in Counter(source_of(r["case"]) for r in flagged).most_common():
        print(f"  {v:5d}  {k}")
    for r in flagged[:15]:
        print(f"    {r['case']}  image_dev={r['image']}  label_dev={r['label']}")

    devs = np.array([r["image"] for r in res
                     if isinstance(r["image"], float) and np.isfinite(r["image"])])
    print(f"\nimage deviation percentiles over clean cases: "
          f"p50={np.percentile(devs,50):.2e} p99={np.percentile(devs,99):.2e} "
          f"max={devs.max():.2e}")
    print(f"wrote {PROJ/'code/nonorthonormal.json'}")


if __name__ == "__main__":
    main()
