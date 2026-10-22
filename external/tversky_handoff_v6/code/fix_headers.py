"""Repair non-orthonormal direction cosines so ITK will read the file.

17 cases carry direction matrices that are off orthonormal by 1e-4..4e-4 -- rounding
noise from whatever wrote them, not real skew. nibabel loads them; ITK refuses
outright, which is what killed nnUNetv2_extract_fingerprint at 92%.

Fix: nearest orthonormal matrix by polar decomposition (SVD, R -> U@Vt), rebuild the
affine with the original spacing and origin, rewrite the voxel data untouched.
Geometrically this moves nothing that matters at 1e-4; it just makes the header
self-consistent.

Corrected files land in _fixed/ and the dataset symlinks are repointed, so the
original download is never modified.
"""
import json
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

PROJ = Path("/media/shashank/NewVolume/PancancerCTSeg")
DST = PROJ / "nnUNet_raw/Dataset501_PanCancer"
FIXED = DST / "_fixed"


def _dev(R):
    """Max |R^T R - I| for a direction matrix whose columns are unit length."""
    return float(np.abs(R.T @ R - np.eye(3)).max())


def orthonormalize(affine):
    """Return (fixed_affine, deviation_before, deviation_after)."""
    A = np.asarray(affine, dtype=np.float64).copy()
    sp = np.linalg.norm(A[:3, :3], axis=0)
    R = A[:3, :3] / sp                       # copy, not a view into A
    before = _dev(R)
    U, _, Vt = np.linalg.svd(R)
    R_orth = U @ Vt
    if np.linalg.det(R_orth) < 0:            # keep handedness
        U[:, -1] *= -1
        R_orth = U @ Vt
    A[:3, :3] = R_orth * sp
    return A, before, _dev(R_orth)


def rewrite(src: Path, dst: Path):
    im = nib.load(str(src))
    new_affine, dev_before, dev_after = orthonormalize(im.affine)
    data = np.asanyarray(im.dataobj)          # scaled read, matches what ITK sees
    hdr = im.header.copy()
    hdr.set_sform(new_affine, code=int(hdr["sform_code"]) or 1)
    hdr.set_qform(new_affine, code=int(hdr["qform_code"]) or 1)
    out = nib.Nifti1Image(data, new_affine, hdr)
    out.set_data_dtype(im.get_data_dtype())
    dst.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(dst))
    return dev_before, dev_after


def main():
    cases = json.loads((PROJ / "code/nonorthonormal.json").read_text())
    if not cases:
        print("nothing to fix")
        return
    print(f"repairing {len(cases)} cases\n")

    import SimpleITK as sitk
    ok = 0
    for cid in cases:
        pairs = [(DST / f"imagesTr/{cid}_0000.nii.gz", FIXED / f"imagesTr/{cid}_0000.nii.gz"),
                 (DST / f"labelsTr/{cid}.nii.gz",      FIXED / f"labelsTr/{cid}.nii.gz")]
        devs = []
        for src, dst in pairs:
            real = Path(src).resolve()
            devs.append(rewrite(real, dst))
        # repoint the dataset symlinks at the corrected copies
        for src, dst in pairs:
            if src.is_symlink() or src.exists():
                src.unlink()
            src.symlink_to(dst.resolve())
        # prove ITK accepts them now
        try:
            for _, dst in pairs:
                sitk.ReadImage(str(dst))
            ok += 1
            status = "OK"
        except Exception as e:
            status = f"STILL FAILS: {str(e)[:80]}"
        print(f"  {cid:44s} img dev {devs[0][0]:.1e}->{devs[0][1]:.1e}  "
              f"lab dev {devs[1][0]:.1e}->{devs[1][1]:.1e}  {status}")

    print(f"\n{ok}/{len(cases)} now readable by SimpleITK")
    if ok != len(cases):
        sys.exit("some cases still unreadable -- inspect before continuing")


if __name__ == "__main__":
    main()
