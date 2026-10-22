#!/usr/bin/env python3
"""Screen one conservative, DSC-first addition to the immutable GF champion.

The component-box STU-Net supplies the contour.  A whole 6-connected component
is added only when it is absent from the champion, is at least 3,000 mm3, at
least half of it is supported by the matched global-box STU-Net, and at least
10% is supported by the independent FreeTumor model.  The fixed public 50 is
the only source of labels; this script cannot create a hidden submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cc3d
import nibabel as nib
import numpy as np


CONNECTIVITY = 6
MINIMUM_COMPONENT_MM3 = 3000.0
PEER_SUPPORT_FRACTION = 0.50
INDEPENDENT_SUPPORT_FRACTION = 0.10
EXPECTED_ANCHOR_DSC = 0.6004316379500749
DSC_GATE = 0.6154316379500749


@dataclass(frozen=True)
class CaseCounts:
    anchor_dsc: float
    candidate_dsc: float
    selected_components: int
    true_selected_components: int
    selected_voxels: int
    selected_true_positive_voxels: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def foreground_extent(mask: np.ndarray) -> tuple[slice, slice, slice] | None:
    coordinates = np.nonzero(mask)
    if not coordinates[0].size:
        return None
    return tuple(slice(int(axis.min()), int(axis.max()) + 1) for axis in coordinates)


def load_binary(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(path))
    values = np.asanyarray(image.dataobj)
    observed = np.unique(values)
    if not np.isin(observed, (0, 1)).all():
        raise ValueError(f"nonbinary mask {path}: {observed[:20].tolist()}")
    return image, values.astype(bool, copy=False)


def validate_geometry(name: str, reference: nib.Nifti1Image, image: nib.Nifti1Image) -> None:
    if reference.shape != image.shape or not np.allclose(
        reference.affine, image.affine, atol=1e-4
    ):
        raise ValueError(f"geometry mismatch for {name}")


def dice_from_counts(tp: int, prediction: int, target: int) -> float:
    denominator = prediction + target
    return float(2 * tp / denominator) if denominator else 1.0


def counts_from_statistics(
    *,
    sizes: np.ndarray,
    anchor_overlap: np.ndarray,
    peer_overlap: np.ndarray,
    independent_overlap: np.ndarray,
    target_overlap: np.ndarray,
    voxel_mm3: float,
    anchor_voxels: int,
    target_voxels: int,
    anchor_tp: int,
) -> CaseCounts:
    eligible = (anchor_overlap == 0) & (
        sizes.astype(np.float64) * voxel_mm3 >= MINIMUM_COMPONENT_MM3
    )
    eligible[0] = False
    peer_fraction = np.divide(
        peer_overlap, sizes, out=np.zeros_like(sizes, dtype=np.float64), where=sizes > 0
    )
    independent_fraction = np.divide(
        independent_overlap,
        sizes,
        out=np.zeros_like(sizes, dtype=np.float64),
        where=sizes > 0,
    )
    keep = eligible & (peer_fraction >= PEER_SUPPORT_FRACTION) & (
        independent_fraction >= INDEPENDENT_SUPPORT_FRACTION
    )
    ids = np.flatnonzero(keep)
    addition_voxels = int(sizes[ids].sum())
    addition_tp = int(target_overlap[ids].sum())
    return CaseCounts(
        anchor_dsc=dice_from_counts(anchor_tp, anchor_voxels, target_voxels),
        candidate_dsc=dice_from_counts(
            anchor_tp + addition_tp,
            anchor_voxels + addition_voxels,
            target_voxels,
        ),
        selected_components=int(ids.size),
        true_selected_components=int(np.count_nonzero(target_overlap[ids])),
        selected_voxels=addition_voxels,
        selected_true_positive_voxels=addition_tp,
    )


def case_counts_from_arrays(
    *,
    proposal: np.ndarray,
    anchor: np.ndarray,
    peer: np.ndarray,
    independent: np.ndarray,
    target: np.ndarray,
    voxel_mm3: float,
) -> CaseCounts:
    if not (proposal.shape == anchor.shape == peer.shape == independent.shape == target.shape):
        raise ValueError("mask shape mismatch")
    extent = foreground_extent(proposal)
    anchor_voxels = int(anchor.sum())
    target_voxels = int(target.sum())
    anchor_tp = int(np.count_nonzero(anchor & target))
    if extent is None:
        sizes = np.zeros(1, dtype=np.int64)
        overlaps = [np.zeros(1, dtype=np.int64) for _ in range(4)]
    else:
        labels, count = cc3d.connected_components(
            np.ascontiguousarray(proposal[extent], dtype=np.uint8),
            connectivity=CONNECTIVITY,
            return_N=True,
            out_dtype=np.uint32,
            binary_image=True,
        )
        if count >= np.iinfo(np.uint32).max:
            raise ValueError("proposal component count exceeds uint32 contract")
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        overlaps = [
            np.bincount(labels[mask[extent]], minlength=count + 1)
            for mask in (anchor, peer, independent, target)
        ]
    return counts_from_statistics(
        sizes=sizes,
        anchor_overlap=overlaps[0],
        peer_overlap=overlaps[1],
        independent_overlap=overlaps[2],
        target_overlap=overlaps[3],
        voxel_mm3=voxel_mm3,
        anchor_voxels=anchor_voxels,
        target_voxels=target_voxels,
        anchor_tp=anchor_tp,
    )


def case_counts_from_paths(
    *, proposal_path: Path, anchor_path: Path, peer_path: Path,
    independent_path: Path, target_path: Path,
) -> CaseCounts:
    proposal_image, proposal = load_binary(proposal_path)
    extent = foreground_extent(proposal)
    voxel_mm3 = float(abs(np.linalg.det(proposal_image.affine[:3, :3])))
    if extent is None:
        sizes = np.zeros(1, dtype=np.int64)
        labels = None
        count = 0
    else:
        labels, count = cc3d.connected_components(
            np.ascontiguousarray(proposal[extent], dtype=np.uint8), connectivity=CONNECTIVITY,
            return_N=True, out_dtype=np.uint32, binary_image=True,
        )
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
    del proposal

    def overlap(path: Path, role: str) -> tuple[np.ndarray, int, np.ndarray | None]:
        image, mask = load_binary(path)
        validate_geometry(f"{role}/{path.name}", proposal_image, image)
        values = (
            np.bincount(labels[mask[extent]], minlength=count + 1)
            if extent is not None and labels is not None
            else np.zeros(1, dtype=np.int64)
        )
        return values, int(mask.sum()), mask

    anchor_overlap, anchor_voxels, anchor = overlap(anchor_path, "anchor")
    target_overlap, target_voxels, target = overlap(target_path, "target")
    anchor_tp = int(np.count_nonzero(anchor & target))
    del anchor, target
    peer_overlap, _, peer = overlap(peer_path, "peer")
    del peer
    independent_overlap, _, independent = overlap(independent_path, "independent")
    del independent, labels
    return counts_from_statistics(
        sizes=sizes,
        anchor_overlap=anchor_overlap,
        peer_overlap=peer_overlap,
        independent_overlap=independent_overlap,
        target_overlap=target_overlap,
        voxel_mm3=voxel_mm3,
        anchor_voxels=anchor_voxels,
        target_voxels=target_voxels,
        anchor_tp=anchor_tp,
    )


def inventory(directory: Path) -> set[str]:
    return {path.name for path in directory.glob("*.nii.gz")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--peer", type=Path, required=True)
    parser.add_argument("--independent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    names = sorted(inventory(args.labels))
    if len(names) != 50:
        raise ValueError(f"fixed-public inventory must contain 50 cases, got {len(names)}")
    expected = set(names)
    for role, directory in (
        ("anchor", args.anchor), ("proposal", args.proposal), ("peer", args.peer),
        ("independent", args.independent),
    ):
        observed = inventory(directory)
        if observed != expected:
            raise ValueError(
                f"{role} inventory mismatch: missing={sorted(expected-observed)} "
                f"extra={sorted(observed-expected)}"
            )

    records = []
    for index, name in enumerate(names, 1):
        counts = case_counts_from_paths(
            proposal_path=args.proposal / name,
            anchor_path=args.anchor / name,
            peer_path=args.peer / name,
            independent_path=args.independent / name,
            target_path=args.labels / name,
        )
        records.append({"case": name.removesuffix(".nii.gz"), **counts.__dict__})
        print(f"[{index:02d}/50] {name}", flush=True)

    anchor_dsc = float(np.mean([row["anchor_dsc"] for row in records]))
    if not np.isclose(anchor_dsc, EXPECTED_ANCHOR_DSC, atol=1e-12, rtol=0):
        raise ValueError(f"anchor DSC identity mismatch: {anchor_dsc} != {EXPECTED_ANCHOR_DSC}")
    candidate_dsc = float(np.mean([row["candidate_dsc"] for row in records]))
    configuration = {
        "configuration": "component_shape_base50_freetumor10_mm3000",
        "Lesion_DSC": candidate_dsc,
        "DSC_gain": candidate_dsc - anchor_dsc,
        "selected_components": int(sum(row["selected_components"] for row in records)),
        "true_selected_components": int(sum(row["true_selected_components"] for row in records)),
        "selected_voxels": int(sum(row["selected_voxels"] for row in records)),
        "selected_true_positive_voxels": int(
            sum(row["selected_true_positive_voxels"] for row in records)
        ),
        "passes_dsc_gate": candidate_dsc >= DSC_GATE,
    }
    payload = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "decision": "QUALIFY_EXACT_NSD_MATERIALIZATION" if configuration["passes_dsc_gate"] else "KEEP_TOUCHCOMP",
        "selection_policy": "single predeclared conservative whole-component addition; no sweep",
        "connectivity": CONNECTIVITY,
        "minimum_component_mm3": MINIMUM_COMPONENT_MM3,
        "minimum_peer_support_fraction": PEER_SUPPORT_FRACTION,
        "minimum_independent_support_fraction": INDEPENDENT_SUPPORT_FRACTION,
        "anchor_DSC": anchor_dsc,
        "fixed_public_DSC_gate": DSC_GATE,
        "configuration": configuration,
        "cases": records,
        "assets": {
            role: {"path": str(path.resolve())}
            for role, path in (
                ("labels", args.labels), ("anchor", args.anchor), ("proposal", args.proposal),
                ("peer", args.peer), ("independent", args.independent),
            )
        },
        "implementation": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "public_read": True,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    atomic_json(args.output, payload)
    print(json.dumps({key: payload[key] for key in ("decision", "anchor_DSC", "configuration")}, indent=2))


if __name__ == "__main__":
    main()
