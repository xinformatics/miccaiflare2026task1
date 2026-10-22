"""Pre-flight integrity scan over all 17,575 image/label pairs.

nnU-Net's own --verify_dataset_integrity reads every image in full (~800GB, hours).
This does the same checks in minutes by reading image HEADERS only (nibabel is lazy)
and loading just the labels, which are small.

Catches, per case:
  missing/zero-byte file, unreadable header, image<->label shape mismatch,
  image<->label affine mismatch, malformed orientation (singular/NaN affine),
  non-finite or non-positive spacing, absurd spacing, label values outside {0,1},
  empty (all-background) labels.

Writes integrity_report.json + excluded_cases.json. Cases in excluded_cases.json are
hard failures that would break preprocessing; empty-label cases are reported
separately since they are usable as training negatives, not errors.
"""
import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from common import source_of  # noqa: E402

PROJ = Path("/media/shashank/NewVolume/PancancerCTSeg")
DST = PROJ / "nnUNet_raw/Dataset501_PanCancer"
IMG, LAB = DST / "imagesTr", DST / "labelsTr"
OUT = PROJ / "code"

# A CT voxel outside this range (mm) is almost certainly a broken header.
SPACING_MIN, SPACING_MAX = 0.05, 20.0
AFFINE_TOL = 1e-3


def check(cid):
    r = {"case": cid, "source": source_of(cid), "errors": [], "warnings": []}
    ip, lp = IMG / f"{cid}_0000.nii.gz", LAB / f"{cid}.nii.gz"
    try:
        for p, tag in ((ip, "image"), (lp, "label")):
            if not p.exists():
                r["errors"].append(f"{tag}_missing")
            elif p.stat().st_size == 0:
                r["errors"].append(f"{tag}_zero_byte")
        if r["errors"]:
            return r

        im = nib.load(str(ip))          # lazy: header only
        lb = nib.load(str(lp))
        ish, lsh = im.shape[:3], lb.shape[:3]
        iaf, laf = im.affine, lb.affine
        izoom = np.asarray(im.header.get_zooms()[:3], dtype=np.float64)

        r.update(shape=list(map(int, ish)),
                 spacing=[round(float(z), 4) for z in izoom],
                 dtype=str(im.get_data_dtype()),
                 axcodes="".join(nib.aff2axcodes(iaf)))

        if len(im.shape) != 3:
            r["errors"].append(f"image_not_3d:{im.shape}")
        if ish != lsh:
            r["errors"].append(f"shape_mismatch:img{list(ish)}!=lab{list(lsh)}")

        if not np.all(np.isfinite(iaf)):
            r["errors"].append("image_affine_nonfinite")
        elif abs(np.linalg.det(iaf[:3, :3])) < 1e-9:
            r["errors"].append("image_affine_singular")

        # ITK refuses non-orthonormal direction cosines outright, and nnU-Net reads
        # through SimpleITK. nibabel is permissive, so this must be checked
        # explicitly -- 17 cases (deviation 1e-4..4e-4, i.e. rounding noise) killed
        # nnUNetv2_extract_fingerprint at 92% before this check existed.
        # code/fix_headers.py repairs them via polar decomposition.
        for aff, tag in ((iaf, "image"), (laf, "label")):
            if np.all(np.isfinite(aff)):
                sp3 = np.linalg.norm(aff[:3, :3], axis=0)
                if np.all(sp3 > 0):
                    R = aff[:3, :3] / sp3
                    dev = float(np.abs(R.T @ R - np.eye(3)).max())
                    if dev > 1e-6:
                        r["errors"].append(f"{tag}_not_orthonormal:{dev:.3g}")
        if not np.all(np.isfinite(laf)):
            r["errors"].append("label_affine_nonfinite")

        if np.all(np.isfinite(iaf)) and np.all(np.isfinite(laf)):
            if not np.allclose(iaf, laf, atol=AFFINE_TOL, rtol=0):
                d = float(np.abs(iaf - laf).max())
                # direction/spacing disagreement is fatal; pure origin shift is
                # recoverable because nnU-Net resamples on voxel grids.
                if not np.allclose(iaf[:3, :3], laf[:3, :3], atol=AFFINE_TOL, rtol=0):
                    r["warnings"].append(f"affine_direction_mismatch:{d:.4g}")
                else:
                    r["warnings"].append(f"affine_origin_shift:{d:.4g}")

        if not np.all(np.isfinite(izoom)) or np.any(izoom <= 0):
            r["errors"].append(f"bad_spacing:{izoom.tolist()}")
        elif izoom.min() < SPACING_MIN or izoom.max() > SPACING_MAX:
            r["errors"].append(f"absurd_spacing:{izoom.tolist()}")

        # A label header that disagrees with the image is only a WARNING: nnU-Net
        # asserts shape equality between image and seg and then uses the image's
        # spacing for everything (default_preprocessor.py:51). The seg affine is
        # never consulted, so a reset/garbage label header is inert as long as the
        # arrays line up -- which was verified by HU statistics inside the mask.
        if "".join(nib.aff2axcodes(iaf)) != "".join(nib.aff2axcodes(laf)):
            r["warnings"].append("orientation_mismatch")

        seg = np.asanyarray(lb.dataobj)
        if not np.all(np.isfinite(seg)):
            r["errors"].append("label_nonfinite")
            return r
        u = np.unique(seg)
        r["label_values"] = [float(x) for x in u[:12]]
        # Compare in float32, which is what nnU-Net's reader (SimpleITKIO) yields.
        # 573 Pancreas_PanTS labels are uint8 0/255 with scl_slope=nan; nibabel's
        # float64 scaling surfaces 1.0000000591389835, but that value is exactly 1.0
        # once narrowed to float32, so nnU-Net sees a clean binary mask. Comparing in
        # float64 here flagged all 573 as corrupt -- a false positive.
        u32 = np.unique(np.asarray(u, dtype=np.float32))
        if not np.all(np.isin(u32, np.float32([0, 1]))):
            r["errors"].append(f"label_values_not_binary:{r['label_values']}")
        fg = int((seg > 0).sum())
        r["fg_voxels"] = fg
        r["fg_mm3"] = round(fg * float(np.prod(np.abs(izoom))), 1)
        if fg == 0:
            r["warnings"].append("empty_label")
    except Exception:
        r["errors"].append("exception:" + traceback.format_exc().splitlines()[-1][:200])
    return r


def main():
    cids = sorted(p.name[: -len(".nii.gz")] for p in LAB.iterdir()
                  if p.name.endswith(".nii.gz"))
    print(f"scanning {len(cids)} cases with {os.cpu_count()} workers ...", flush=True)

    res = []
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 2)) as ex:
        for i, r in enumerate(ex.map(check, cids, chunksize=16), 1):
            res.append(r)
            if i % 1000 == 0:
                print(f"  {i}/{len(cids)}", flush=True)

    bad = [r for r in res if r["errors"]]
    warn = [r for r in res if r["warnings"] and not r["errors"]]
    empty = [r for r in res if "empty_label" in r["warnings"]]

    (OUT / "integrity_report.json").write_text(json.dumps(res, indent=1))
    (OUT / "excluded_cases.json").write_text(
        json.dumps(sorted(r["case"] for r in bad), indent=1))

    print(f"\n================ INTEGRITY: {len(res)} cases ================")
    print(f"HARD FAILURES : {len(bad)}")
    print(f"warnings only : {len(warn)}")
    print(f"empty labels  : {len(empty)}")

    from collections import Counter
    ec = Counter(e.split(":")[0] for r in bad for e in r["errors"])
    wc = Counter(w.split(":")[0] for r in res for w in r["warnings"])
    if ec:
        print("\nerror types:")
        for k, v in ec.most_common():
            print(f"  {v:6d}  {k}")
    if wc:
        print("warning types:")
        for k, v in wc.most_common():
            print(f"  {v:6d}  {k}")
    if bad:
        print("\nfailures by source:")
        for k, v in Counter(r["source"] for r in bad).most_common(15):
            print(f"  {v:6d}  {k}")
        print("\nexamples:")
        for r in bad[:8]:
            print(f"  {r['case']}: {r['errors']}")

    ok = [r for r in res if not r["errors"]]
    print(f"\nUSABLE FOR TRAINING: {len(ok)}")
    print(f"  of which empty-label (negatives): {len(empty)}")
    ax = Counter(r.get("axcodes", "?") for r in ok)
    print(f"orientation codes: {dict(ax.most_common(8))}")
    print(f"\nwrote {OUT/'integrity_report.json'} and {OUT/'excluded_cases.json'}")


if __name__ == "__main__":
    main()
