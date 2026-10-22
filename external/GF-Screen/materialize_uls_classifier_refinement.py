#!/usr/bin/env python3
"""Refine grouped-OOF selected components with ULS+ without reading labels."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import cc3d
import nibabel as nib
import numpy as np
import SimpleITK as sitk

from materialize_uls_consensus import boxes_overlap, crop_indices_and_bbox
from run_uls_plus_component_refiner import paste_crop, same_geometry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--component-csv", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("refined", "seed_union"), required=True)
    parser.add_argument("--min-seed-coverage", type=float, required=True)
    parser.add_argument("--min-refined-overlap", type=float, required=True)
    parser.add_argument("--min-volume-ratio", type=float, required=True)
    parser.add_argument("--max-volume-ratio", type=float, required=True)
    parser.add_argument("--agreement-mutual", type=float, default=0.0)
    parser.add_argument("--agreement-min-mm3", type=float, default=500.0)
    args = parser.parse_args()

    selected: dict[str, set[int]] = {}
    with args.component_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["oof_selected"]):
                selected.setdefault(row["case"], set()).add(int(row["component_id"]))

    records = json.loads((args.work / "manifest.json").read_text())
    by_case: dict[str, list[dict]] = {}
    for record in records:
        by_case.setdefault(record["case"], []).append(record)
    args.output.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict[str, object]] = []

    for case, case_records in sorted(by_case.items()):
        proposal_img = sitk.ReadImage(case_records[0]["proposal"])
        proposal = sitk.GetArrayFromImage(proposal_img) > 0
        # The classifier assigned component IDs in nibabel x-y-z array order.
        # Preserve those exact IDs, then transpose to SimpleITK z-y-x order so
        # manifest click indices address the same physical component.
        proposal_xyz = np.asanyarray(
            nib.load(case_records[0]["proposal"]).dataobj
        ).astype(bool)
        component_map_xyz = cc3d.connected_components(
            proposal_xyz.astype(np.uint8), connectivity=26
        )
        component_map = np.transpose(component_map_xyz, (2, 1, 0))
        if component_map.shape != proposal.shape or not np.array_equal(
            component_map > 0, proposal
        ):
            raise RuntimeError(f"component coordinate mismatch for {case}")
        component_ids = selected.get(case, set())
        rescue = np.zeros(proposal.shape, dtype=bool)
        record_by_component: dict[int, dict] = {}
        all_sparse = []

        for record in case_records:
            z, y, x = map(int, record["click_zyx"])
            component_id = int(component_map[z, y, x])
            if component_id:
                record_by_component[component_id] = record
            if args.agreement_mutual > 0:
                path = args.work / "predicted_crops" / f"{record['crop_id']}.nii.gz"
                crop = sitk.GetArrayFromImage(sitk.ReadImage(str(path))) > 0
                indices, bbox = crop_indices_and_bbox(
                    crop, proposal.shape, record["start_xyz"]
                )
                if indices.size and bbox is not None:
                    all_sparse.append((record["crop_id"], indices, bbox))

        for component_id in sorted(component_ids):
            seed = component_map == component_id
            seed_voxels = int(seed.sum())
            record = record_by_component.get(component_id)
            accepted = False
            seed_coverage = refined_overlap = volume_ratio = 0.0
            refined_voxels = 0
            if record is not None:
                path = args.work / "predicted_crops" / f"{record['crop_id']}.nii.gz"
                crop = sitk.GetArrayFromImage(sitk.ReadImage(str(path))) > 0
                refined = paste_crop(crop, proposal.shape, record["start_xyz"])
                refined_voxels = int(refined.sum())
                intersection = int(np.count_nonzero(refined & seed))
                seed_coverage = intersection / max(seed_voxels, 1)
                refined_overlap = intersection / max(refined_voxels, 1)
                volume_ratio = refined_voxels / max(seed_voxels, 1)
                accepted = (
                    refined_voxels > 0
                    and seed_coverage >= args.min_seed_coverage
                    and refined_overlap >= args.min_refined_overlap
                    and args.min_volume_ratio <= volume_ratio <= args.max_volume_ratio
                )
                if accepted:
                    rescue |= refined if args.mode == "refined" else (seed | refined)
                else:
                    rescue |= seed
            else:
                rescue |= seed
            audit_rows.append(
                {
                    "case": case,
                    "source": "classifier",
                    "component_id": component_id,
                    "crop_id": record["crop_id"] if record else "",
                    "accepted": int(accepted),
                    "mutual_dice": "",
                    "seed_voxels": seed_voxels,
                    "refined_voxels": refined_voxels,
                    "seed_coverage": seed_coverage,
                    "refined_overlap": refined_overlap,
                    "volume_ratio": volume_ratio,
                }
            )

        if args.agreement_mutual > 0:
            pair_candidates = []
            voxel_mm3 = float(np.prod(proposal_img.GetSpacing()))
            for (name_a, idx_a, box_a), (name_b, idx_b, box_b) in itertools.combinations(
                all_sparse, 2
            ):
                if not boxes_overlap(box_a, box_b):
                    continue
                intersection = np.intersect1d(idx_a, idx_b, assume_unique=True)
                if intersection.size == 0:
                    continue
                mutual = 2.0 * intersection.size / (idx_a.size + idx_b.size)
                volume = float(intersection.size * voxel_mm3)
                if mutual >= args.agreement_mutual and volume >= args.agreement_min_mm3:
                    pair_candidates.append((mutual, volume, name_a, name_b, intersection))
            pair_candidates.sort(key=lambda row: (-row[0], -row[1], row[2], row[3]))
            if pair_candidates:
                mutual, _, name_a, name_b, intersection = pair_candidates[0]
                rescue.ravel()[intersection] = True
                audit_rows.append(
                    {
                        "case": case,
                        "source": "agreement",
                        "component_id": "",
                        "crop_id": f"{name_a}|{name_b}",
                        "accepted": 1,
                        "mutual_dice": mutual,
                        "seed_voxels": "",
                        "refined_voxels": int(intersection.size),
                        "seed_coverage": "",
                        "refined_overlap": "",
                        "volume_ratio": "",
                    }
                )

        base_img = sitk.ReadImage(str(args.base / f"{case}.nii.gz"))
        if not same_geometry(proposal_img, base_img):
            raise RuntimeError(f"base geometry mismatch for {case}")
        output = (sitk.GetArrayFromImage(base_img) > 0) | rescue
        output_img = sitk.GetImageFromArray(output.astype(np.uint8))
        output_img.CopyInformation(proposal_img)
        sitk.WriteImage(output_img, str(args.output / f"{case}.nii.gz"), True)
        print(
            f"{case}: selected={len(component_ids)} rescue_voxels={int(rescue.sum())}",
            flush=True,
        )

    fields = [
        "case",
        "source",
        "component_id",
        "crop_id",
        "accepted",
        "mutual_dice",
        "seed_voxels",
        "refined_voxels",
        "seed_coverage",
        "refined_overlap",
        "volume_ratio",
    ]
    with (args.output.parent / f"{args.output.name}.audit.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit_rows)


if __name__ == "__main__":
    main()
