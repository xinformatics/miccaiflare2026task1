"""Build the nnU-Net raw dataset as SYMLINKS.

The images are ~800GB; copying them is not an option and not necessary — nnU-Net
only ever reads them. Merges imagesTr_part1/part2 into one imagesTr, likewise for
labels, and writes the dataset.json nnU-Net expects.
"""
import json
import os
from pathlib import Path

PROJ = Path("/media/shashank/NewVolume/PancancerCTSeg")
SRC = PROJ / "train_label"
DST = PROJ / "nnUNet_raw/Dataset501_PanCancer"

IMG_PARTS = ["imagesTr_part1", "imagesTr_part2"]
LAB_PARTS = ["labelsTr_part1", "labelsTr_part2"]


def link_all(parts, dst_dir, suffix_check):
    dst_dir.mkdir(parents=True, exist_ok=True)
    n, skipped = 0, 0
    for part in parts:
        src_dir = SRC / part
        for f in os.scandir(src_dir):
            if not f.name.endswith(".nii.gz"):
                continue
            if suffix_check and not f.name.endswith("_0000.nii.gz"):
                skipped += 1
                continue
            link = dst_dir / f.name
            if link.is_symlink() or link.exists():
                n += 1
                continue
            link.symlink_to(f.path)
            n += 1
    return n, skipped


def main():
    ni, sk = link_all(IMG_PARTS, DST / "imagesTr", suffix_check=True)
    nl, _ = link_all(LAB_PARTS, DST / "labelsTr", suffix_check=False)
    print(f"imagesTr: {ni} symlinks (skipped {sk} non-_0000)")
    print(f"labelsTr: {nl} symlinks")

    imgs = {p.name[: -len("_0000.nii.gz")] for p in (DST / "imagesTr").iterdir()}
    labs = {p.name[: -len(".nii.gz")] for p in (DST / "labelsTr").iterdir()}
    print(f"unique image ids {len(imgs)}, label ids {len(labs)}")
    only_img, only_lab = imgs - labs, labs - imgs
    if only_img or only_lab:
        print(f"!! image-without-label: {len(only_img)} e.g. {list(only_img)[:3]}")
        print(f"!! label-without-image: {len(only_lab)} e.g. {list(only_lab)[:3]}")
    paired = imgs & labs
    print(f"PAIRED CASES: {len(paired)}")

    (DST / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": len(paired),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }, indent=2))
    print(f"wrote {DST/'dataset.json'} (numTraining={len(paired)})")


if __name__ == "__main__":
    main()
