"""Reuse Dataset501's fingerprint/plans for the subset dataset.

The subset's cases are drawn from 501, so its fingerprint would be a strictly worse
estimate of the same distribution -- and re-extracting it means another pass over
hundreds of GB. Copy 501's instead and just retarget the dataset name.

This also keeps the dry run honest: the subset is then preprocessed with exactly the
normalization constants and target spacing the full run will use.

  python code/clone_plans.py --src 501 --src-name PanCancer --dst 502 --dst-name Sample
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from arch import CONFIG_NAME, build_config  # noqa: E402

PRE = Path(os.environ["nnUNet_preprocessed"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=int, default=501)
    ap.add_argument("--src-name", default="PanCancer")
    ap.add_argument("--dst", type=int, default=502)
    ap.add_argument("--dst-name", default="Sample")
    a = ap.parse_args()

    src = PRE / f"Dataset{a.src:03d}_{a.src_name}"
    dst = PRE / f"Dataset{a.dst:03d}_{a.dst_name}"
    dst.mkdir(parents=True, exist_ok=True)

    fp = src / "dataset_fingerprint.json"
    if not fp.exists():
        sys.exit(f"missing {fp} -- run nnUNetv2_extract_fingerprint -d {a.src} first")
    shutil.copy2(fp, dst / "dataset_fingerprint.json")

    plans_p = src / "nnUNetPlans.json"
    if not plans_p.exists():
        sys.exit(f"missing {plans_p} -- run nnUNetv2_plan_experiment -d {a.src} first")
    plans = json.loads(plans_p.read_text())
    plans["dataset_name"] = dst.name
    if CONFIG_NAME not in plans["configurations"]:
        plans["configurations"][CONFIG_NAME] = build_config(
            plans["configurations"]["3d_fullres"])
    (dst / "nnUNetPlans.json").write_text(json.dumps(plans, indent=4))

    # nnU-Net wants dataset.json alongside the plans
    raw_ds = Path(os.environ["nnUNet_raw"]) / dst.name / "dataset.json"
    if raw_ds.exists():
        shutil.copy2(raw_ds, dst / "dataset.json")

    # splits: the dry run trains on everything, no held-out fold needed
    ids = sorted(p.name[: -len(".nii.gz")] for p in
                 (Path(os.environ["nnUNet_raw"]) / dst.name / "labelsTr").iterdir()
                 if p.name.endswith(".nii.gz"))
    (dst / "splits_final.json").write_text(json.dumps(
        [{"train": ids, "val": ids[: max(1, len(ids) // 10)]}], indent=1))

    print(f"cloned fingerprint + plans  {src.name} -> {dst.name}")
    print(f"  config      : {CONFIG_NAME}")
    print(f"  cases       : {len(ids)}")
    print(f"  spacing     : {plans['configurations'][CONFIG_NAME]['spacing']}")
    print(f"  patch       : {plans['configurations'][CONFIG_NAME]['patch_size']}")
    print(f"  fg intensity: {plans['foreground_intensity_properties_per_channel']}")


if __name__ == "__main__":
    main()
