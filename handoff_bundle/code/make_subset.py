"""Build a stratified subset dataset (symlinks) for fast end-to-end validation.

Sampling is proportional-with-a-floor across all 28 source cohorts: DeepLesion's
17-slice volumes and KiTS23's 581-slice volumes have wildly different footprints, so
a naive random sample would mis-measure both preprocessing cost and training speed.

  python code/make_subset.py --n 400 --id 502 --name Sample
"""
import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import source_of  # noqa: E402

PROJ = Path("/media/shashank/NewVolume/PancancerCTSeg")
FULL = PROJ / "nnUNet_raw/Dataset501_PanCancer"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--id", type=int, default=502)
    ap.add_argument("--name", default="Sample")
    ap.add_argument("--min-per-source", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    a = ap.parse_args()

    excl = set()
    ep = PROJ / "code/excluded_cases.json"
    if ep.exists():
        excl = set(json.loads(ep.read_text()))
        print(f"excluding {len(excl)} cases flagged by the integrity scan")

    cids = [p.name[: -len(".nii.gz")] for p in (FULL / "labelsTr").iterdir()
            if p.name.endswith(".nii.gz")]
    cids = [c for c in cids if c not in excl]

    by = defaultdict(list)
    for c in cids:
        by[source_of(c)].append(c)
    for v in by.values():
        v.sort()

    rng = random.Random(a.seed)
    total = len(cids)
    picked = []
    for src, v in sorted(by.items()):
        k = max(a.min_per_source, int(math.ceil(a.n * len(v) / total)))
        k = min(k, len(v))
        picked += rng.sample(v, k)
    picked = sorted(set(picked))

    dst = PROJ / f"nnUNet_raw/Dataset{a.id:03d}_{a.name}"
    for sub in ("imagesTr", "labelsTr"):
        (dst / sub).mkdir(parents=True, exist_ok=True)
        for old in (dst / sub).iterdir():
            old.unlink()
    for c in picked:
        os.symlink(os.path.realpath(FULL / f"imagesTr/{c}_0000.nii.gz"),
                   dst / f"imagesTr/{c}_0000.nii.gz")
        os.symlink(os.path.realpath(FULL / f"labelsTr/{c}.nii.gz"),
                   dst / f"labelsTr/{c}.nii.gz")

    (dst / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": len(picked),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }, indent=2))

    print(f"\n{dst.name}: {len(picked)} cases from {len(by)} sources")
    for src, v in sorted(by.items()):
        got = sum(1 for c in picked if source_of(c) == src)
        print(f"  {src:34s} {got:4d} / {len(v):5d}")
    gb = sum(os.path.getsize(os.path.realpath(dst / f"imagesTr/{c}_0000.nii.gz"))
             for c in picked) / 2**30
    print(f"\nraw image bytes in subset: {gb:.1f} GB "
          f"({gb/len(picked)*1024:.0f} MB/case)")


if __name__ == "__main__":
    main()
