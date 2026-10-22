"""Measure the preprocessed-cache footprint and extrapolate to the full dataset.

The go/no-go on full-data training: ~127GB free on the data volume vs 17,575 cases.
nnU-Net 2.6 writes blosc2 .b2nd (ZSTD, memory-mapped, no unpack step), so the real
number is far below the float32 arithmetic -- but measure it, don't assume it.
"""
import argparse
import os
from pathlib import Path

PROJ = Path("/media/shashank/NewVolume/PancancerCTSeg")
FULL_N = 17575


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="502")
    ap.add_argument("--name", default="Sample")
    ap.add_argument("--config", default="3d_fullres_S4D2W32")
    a = ap.parse_args()

    d = (Path(os.environ["nnUNet_preprocessed"]) /
         f"Dataset{int(a.dataset):03d}_{a.name}" / f"nnUNetPlans_{a.config}")
    if not d.exists():
        raise SystemExit(f"not found: {d}\n(has preprocessing run for this config?)")

    data = sorted(p for p in d.iterdir() if p.name.endswith(".b2nd")
                  and not p.name.endswith("_seg.b2nd"))
    segs = sorted(p for p in d.iterdir() if p.name.endswith("_seg.b2nd"))
    pkls = sorted(p for p in d.iterdir() if p.name.endswith(".pkl"))

    def tot(ps):
        return sum(p.stat().st_size for p in ps)

    n = len(data)
    if n == 0:
        raise SystemExit(f"no .b2nd files in {d}")
    tb, ts, tp = tot(data), tot(segs), tot(pkls)
    total = tb + ts + tp
    per = total / n

    sizes = sorted(p.stat().st_size for p in data)
    import numpy as np
    q = np.percentile(sizes, [50, 90, 99, 100]) / 2**20

    print(f"preprocessed dir : {d}")
    print(f"cases            : {n}")
    print(f"  data  .b2nd    : {tb/2**30:8.2f} GB  ({tb/n/2**20:7.2f} MB/case)")
    print(f"  seg   .b2nd    : {ts/2**30:8.2f} GB  ({ts/n/2**20:7.2f} MB/case)")
    print(f"  props .pkl     : {tp/2**30:8.2f} GB")
    print(f"  TOTAL          : {total/2**30:8.2f} GB  ({per/2**20:7.2f} MB/case)")
    # the tail, not the mean, is what drives peak RAM during preprocessing
    print(f"  data/case MB   : p50 {q[0]:.2f}  p90 {q[1]:.2f}  "
          f"p99 {q[2]:.2f}  MAX {q[3]:.2f}")

    proj = per * FULL_N / 2**30
    # check the filesystem the cache actually lives on, not the project dir --
    # nnUNet_preprocessed is on the HDD while the repo is on the NVMe
    st = os.statvfs(d)
    free = st.f_bavail * st.f_frsize / 2**30
    print(f"\nextrapolated to {FULL_N} cases: {proj:.1f} GB")
    print(f"free where the cache lives ({d}): {free:.1f} GB")
    verdict = "FITS" if proj < free * 0.85 else "DOES NOT FIT"
    print(f"VERDICT: {verdict}  (using 85% of free space as the safety line)")
    if verdict != "FITS":
        print("\nLever: coarsen target spacing in code/arch.py (SPACING).")
        print("Val profile shows p0 lesion = 2606 mm^3, so 2.5x2.5x3mm still leaves")
        print("the smallest lesion >150 voxels. Do not drop anatomy -- it is a")
        print("whole-body task and every source is in-domain.")


if __name__ == "__main__":
    main()
