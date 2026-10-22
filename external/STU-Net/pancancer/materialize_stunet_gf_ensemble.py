#!/usr/bin/env python3
"""Replay one fixed-public-selected STU-Net component ensemble label-blind."""

from __future__ import annotations

import argparse
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import nibabel as nib
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_stunet_gf_ensemble import (  # noqa: E402
    confirmed_anchor_components,
    load_compact_probability,
    probability_name,
    selected_components,
)


def parse_configuration(
    configuration: str,
) -> tuple[bool, int | None, float | None, float | None, str | None]:
    confirmed = re.fullmatch(
        r"gf_anchor_confirm_(stu|ftu|either|both)_any", configuration
    )
    if confirmed:
        return False, None, None, None, confirmed.group(1)
    confirmed_probability = re.fullmatch(
        r"gf_anchor_confirm_either_p(70)", configuration
    )
    if confirmed_probability:
        return (
            False,
            None,
            None,
            int(confirmed_probability.group(1)) / 100.0,
            "either",
        )
    if configuration == "stunet_standalone":
        return False, None, None, None, None
    calibrated = re.fullmatch(r"stunet_p(30|40|60|70)_mm1000", configuration)
    if calibrated:
        return False, 1000, None, int(calibrated.group(1)) / 100.0, None
    calibrated_union = re.fullmatch(
        r"gf_union_stu_p(30|40|60|70)_mm3000", configuration
    )
    if calibrated_union:
        return True, 3000, None, int(calibrated_union.group(1)) / 100.0, None
    calibrated_supported = re.fullmatch(
        r"gf_union_stu_p(30|40)_ftu_s10_mm3000", configuration
    )
    if calibrated_supported:
        return True, 3000, 0.10, int(calibrated_supported.group(1)) / 100.0, None
    standalone = re.fullmatch(r"stunet_mm(500|1000|3000)", configuration)
    if standalone:
        return False, int(standalone.group(1)), None, None, None
    standalone_supported = re.fullmatch(
        r"stunet_ftu_s(01|10)_mm(500|1000|3000)", configuration
    )
    if standalone_supported:
        return (
            False,
            int(standalone_supported.group(2)),
            int(standalone_supported.group(1)) / 100.0,
            None,
            None,
        )
    direct = re.fullmatch(r"gf_union_stu_mm(500|1000|3000)", configuration)
    if direct:
        return True, int(direct.group(1)), None, None, None
    supported = re.fullmatch(
        r"gf_union_stu_ftu_s(01|10)_mm(500|1000|3000)", configuration
    )
    if supported:
        return True, int(supported.group(2)), int(supported.group(1)) / 100.0, None, None
    raise ValueError(f"invalid STU-Net ensemble configuration: {configuration}")


def load_binary(path: Path, reference: nib.Nifti1Image | None = None):
    image = nib.load(str(path))
    if reference is not None and (
        image.shape != reference.shape
        or not np.allclose(image.affine, reference.affine, atol=1e-4)
    ):
        raise ValueError(f"geometry mismatch: {path}")
    values = np.asanyarray(image.dataobj)
    if not np.isin(np.unique(values), [0, 1]).all():
        raise ValueError(f"nonbinary mask: {path}")
    return image, values.astype(bool, copy=False)


def materialize_case(
    name: str,
    anchor_dir: Path,
    stunet_dir: Path,
    freetumor_dir: Path | None,
    use_anchor: bool,
    minimum_mm3: int | None,
    support_fraction: float | None,
    probability_threshold: float | None,
    confirmation_mode: str | None,
    output: Path,
) -> str:
    anchor_image, anchor = load_binary(anchor_dir / name)
    _, stunet = load_binary(stunet_dir / name, anchor_image)
    support = None
    if support_fraction is not None or confirmation_mode in ("ftu", "either", "both"):
        if freetumor_dir is None:
            raise ValueError("FreeTumor support directory is required")
        _, support = load_binary(freetumor_dir / name, anchor_image)
    if confirmation_mode is not None:
        confirmation_stunet = (
            load_compact_probability(
                stunet_dir / probability_name(name), anchor_image.shape
            ) > probability_threshold
            if probability_threshold is not None
            else stunet
        )
        mask = confirmed_anchor_components(
            anchor,
            confirmation_stunet,
            support if support is not None else np.zeros_like(anchor),
            confirmation_mode,
        )
        header = anchor_image.header.copy();header.set_data_dtype(np.uint8)
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), anchor_image.affine, header), output / name)
        return name
    voxel_mm3 = float(abs(np.linalg.det(anchor_image.affine[:3, :3])))
    proposal = (
        load_compact_probability(
            stunet_dir / probability_name(name), anchor_image.shape
        ) > probability_threshold
        if probability_threshold is not None
        else stunet
    )
    selected = (
        proposal
        if minimum_mm3 is None
        else selected_components(
            proposal,
            voxel_mm3,
            minimum_mm3,
            support=support,
            minimum_support_fraction=support_fraction or 0.0,
        )
    )
    mask = (anchor | selected) if use_anchor else selected
    header = anchor_image.header.copy();header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), anchor_image.affine, header), output / name)
    return name


def materialize(
    anchor_dir: Path,
    stunet_dir: Path,
    freetumor_dir: Path | None,
    configuration: str,
    output: Path,
    workers: int,
) -> list[str]:
    use_anchor, minimum, fraction, probability_threshold, confirmation_mode = parse_configuration(configuration)
    names = sorted(path.name for path in anchor_dir.glob("*.nii.gz"))
    expected = set(names)
    if not names or {path.name for path in stunet_dir.glob("*.nii.gz")} != expected:
        raise ValueError("anchor/STU-Net inventory mismatch")
    needs_freetumor = fraction is not None or confirmation_mode in ("ftu", "either", "both")
    if needs_freetumor and (
        freetumor_dir is None
        or {path.name for path in freetumor_dir.glob("*.nii.gz")} != expected
    ):
        raise ValueError("FreeTumor support inventory mismatch")
    output.mkdir(parents=True, exist_ok=True)
    operation = partial(
        materialize_case,
        anchor_dir=anchor_dir,
        stunet_dir=stunet_dir,
        freetumor_dir=freetumor_dir,
        use_anchor=use_anchor,
        minimum_mm3=minimum,
        support_fraction=fraction,
        probability_threshold=probability_threshold,
        confirmation_mode=confirmation_mode,
        output=output,
    )
    if workers == 1:
        return [operation(name) for name in names]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(operation, names))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", required=True, type=Path)
    parser.add_argument("--stunet", required=True, type=Path)
    parser.add_argument("--freetumor", type=Path)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", default=4, type=int)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    names = materialize(
        args.anchor,
        args.stunet,
        args.freetumor,
        args.configuration,
        args.output,
        args.workers,
    )
    print(f"materialized {len(names)} cases with {args.configuration}")


if __name__ == "__main__":
    main()
