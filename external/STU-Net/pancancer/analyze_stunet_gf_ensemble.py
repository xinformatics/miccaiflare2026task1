#!/usr/bin/env python3
"""Predeclared component-level STU-Net/FreeTumor/GF ensemble screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "external" / "GF-Screen"))
from fast_official_lesion_metrics import (  # noqa: E402
    prepare_surface_reference,
    score_case_with_reference,
)


MINIMUM_COMPONENT_MM3 = (500, 1000, 3000)
SUPPORT_FRACTIONS = (0.01, 0.10)
PROBABILITY_THRESHOLDS = (0.30, 0.40, 0.60, 0.70)
ANCHOR_CONFIRMATION_PROBABILITY_THRESHOLDS = (0.70,)
DSC_GATE = 0.6154316379500749
NSD_FLOOR = 0.5104225107769019
HIGH_VALUE_MM3 = 3000.0
DETECTION_FRACTION = 0.10


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def case_cache_path(cache_dir: Path, case_name: str) -> Path:
    return cache_dir / f"{case_name.removesuffix('.nii.gz')}.metrics.json"


def case_source_signature(
    item: tuple[str, str, str, str, str, str | None, bool],
) -> dict[str, dict[str, int | str]]:
    """Cheaply bind a case cache to every file that affects its metrics."""

    name, labels_dir, anchor_dir, stunet_dir, freetumor_dir, probability_dir, closed_only = item
    paths = {
        "label": Path(labels_dir) / name,
        "anchor": Path(anchor_dir) / name,
        "stunet": Path(stunet_dir) / name,
        "freetumor": Path(freetumor_dir) / name,
    }
    if probability_dir is not None:
        paths["stunet_probability"] = Path(probability_dir) / probability_name(name)
    signature = {}
    for role, path in paths.items():
        stat = path.stat()
        signature[role] = {
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    signature["analysis_mode"] = {"closed_only": closed_only}
    return signature


def load_case_cache(
    path: Path,
    case_name: str,
    source_signature: dict[str, dict[str, int | str]],
) -> tuple[str, dict[str, tuple[float, float]], dict] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != 1
            or payload.get("status") != "COMPLETE"
            or payload.get("case") != case_name
            or payload.get("source_signature") != source_signature
            or not isinstance(payload.get("metrics"), dict)
            or not isinstance(payload.get("diagnostic"), dict)
        ):
            return None
        metrics = {}
        for key, value in payload["metrics"].items():
            if not isinstance(key, str) or not isinstance(value, list) or len(value) != 2:
                return None
            pair = tuple(float(number) for number in value)
            if not np.isfinite(pair).all():
                return None
            metrics[key] = pair
        if not metrics:
            return None
        return case_name, metrics, payload["diagnostic"]
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def write_case_cache(
    path: Path,
    result: tuple[str, dict[str, tuple[float, float]], dict],
    source_signature: dict[str, dict[str, int | str]],
) -> None:
    case_name, metrics, diagnostic = result
    atomic_json(
        path,
        {
            "schema_version": 1,
            "status": "COMPLETE",
            "case": case_name,
            "source_signature": source_signature,
            "metrics": {key: list(value) for key, value in metrics.items()},
            "diagnostic": diagnostic,
            "public_labels_read": True,
            "hidden_read": False,
            "submission_created": False,
            "uploaded": False,
        },
    )


def selected_components(
    candidate: np.ndarray,
    voxel_mm3: float,
    minimum_mm3: float,
    support: np.ndarray | None = None,
    minimum_support_fraction: float = 0.0,
) -> np.ndarray:
    labels, count = ndimage.label(
        candidate.astype(bool), structure=ndimage.generate_binary_structure(3, 1)
    )
    if count == 0:
        return np.zeros_like(candidate, dtype=bool)
    sizes = np.bincount(labels.ravel()).astype(float)
    keep = sizes * voxel_mm3 >= minimum_mm3
    if support is not None:
        overlap = np.bincount(
            labels.ravel(), weights=support.astype(bool).ravel(), minlength=count + 1
        )
        fractions = np.divide(overlap, sizes, out=np.zeros_like(overlap), where=sizes > 0)
        keep &= fractions >= minimum_support_fraction
    keep[0] = False
    return keep[labels]


def confirmed_anchor_components(
    anchor: np.ndarray,
    stunet: np.ndarray,
    freetumor: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Keep whole 26-connected anchor components confirmed by independent masks."""

    labels, count = ndimage.label(anchor.astype(bool), structure=np.ones((3, 3, 3)))
    if count == 0:
        return np.zeros_like(anchor, dtype=bool)
    stunet_ids = np.zeros(count + 1, dtype=bool)
    freetumor_ids = np.zeros(count + 1, dtype=bool)
    stunet_ids[np.unique(labels[anchor & stunet])] = True
    freetumor_ids[np.unique(labels[anchor & freetumor])] = True
    if mode == "stu":
        keep = stunet_ids
    elif mode == "ftu":
        keep = freetumor_ids
    elif mode == "either":
        keep = stunet_ids | freetumor_ids
    elif mode == "both":
        keep = stunet_ids & freetumor_ids
    else:
        raise ValueError(f"invalid anchor confirmation mode: {mode}")
    keep[0] = False
    return keep[labels]


def configurations(
    anchor: np.ndarray,
    stunet: np.ndarray,
    freetumor: np.ndarray,
    voxel_mm3: float,
    lesion_probability: np.ndarray | None = None,
    closed_only: bool = False,
) -> dict[str, np.ndarray]:
    masks = {"anchor": anchor, "stunet_standalone": stunet}
    if closed_only:
        if lesion_probability is None:
            raise ValueError("closed-only analysis requires lesion probabilities")
        proposal = lesion_probability > 0.70
        masks["gf_anchor_confirm_either_p70"] = confirmed_anchor_components(
            anchor, proposal, freetumor, "either"
        )
        return masks
    for mode in ("stu", "ftu", "either", "both"):
        masks[f"gf_anchor_confirm_{mode}_any"] = confirmed_anchor_components(
            anchor, stunet, freetumor, mode
        )
    for minimum in MINIMUM_COMPONENT_MM3:
        masks[f"stunet_mm{minimum}"] = selected_components(
            stunet, voxel_mm3, minimum
        )
        masks[f"gf_union_stu_mm{minimum}"] = anchor | selected_components(
            stunet, voxel_mm3, minimum
        )
    for fraction in SUPPORT_FRACTIONS:
        fraction_tag = int(round(fraction * 100))
        for minimum in MINIMUM_COMPONENT_MM3:
            supported = selected_components(
                stunet,
                voxel_mm3,
                minimum,
                support=freetumor,
                minimum_support_fraction=fraction,
            )
            masks[f"stunet_ftu_s{fraction_tag:02d}_mm{minimum}"] = supported
            masks[f"gf_union_stu_ftu_s{fraction_tag:02d}_mm{minimum}"] = (
                anchor | supported
            )
    if lesion_probability is not None:
        if lesion_probability.shape != stunet.shape:
            raise ValueError("probability/mask shape mismatch")
        for threshold in PROBABILITY_THRESHOLDS:
            tag = int(round(threshold * 100))
            proposal = lesion_probability > threshold
            masks[f"stunet_p{tag:02d}_mm1000"] = selected_components(
                proposal, voxel_mm3, 1000
            )
            masks[f"gf_union_stu_p{tag:02d}_mm3000"] = (
                anchor | selected_components(proposal, voxel_mm3, 3000)
            )
            if threshold in (0.30, 0.40):
                masks[f"gf_union_stu_p{tag:02d}_ftu_s10_mm3000"] = (
                    anchor
                    | selected_components(
                        proposal,
                        voxel_mm3,
                        3000,
                        support=freetumor,
                        minimum_support_fraction=0.10,
                    )
                )
            if threshold in ANCHOR_CONFIRMATION_PROBABILITY_THRESHOLDS:
                masks[f"gf_anchor_confirm_either_p{tag:02d}"] = (
                    confirmed_anchor_components(
                        anchor, proposal, freetumor, "either"
                    )
                )
    return masks


def probability_name(case_name: str) -> str:
    return case_name.removesuffix(".nii.gz") + ".prob16.npz"


def load_compact_probability(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    with np.load(path) as archive:
        quantized = archive["lesion_probability_uint16"]
    if quantized.dtype != np.uint16 or quantized.shape != shape:
        raise ValueError(f"invalid compact probability: {path}")
    return quantized.astype(np.float32) / 65535.0


def high_value_diagnostic(
    gt: np.ndarray,
    anchor: np.ndarray,
    candidates: dict[str, np.ndarray],
    voxel_mm3: float,
) -> dict:
    structure = ndimage.generate_binary_structure(3, 1)
    gt_labels, gt_count = ndimage.label(gt.astype(bool), structure=structure)
    missed = []
    for component in range(1, gt_count + 1):
        lesion = gt_labels == component
        voxels = int(lesion.sum())
        if voxels * voxel_mm3 < HIGH_VALUE_MM3:
            continue
        if np.count_nonzero(anchor & lesion) / voxels >= DETECTION_FRACTION:
            continue
        missed.append(
            {
                key: bool(
                    np.count_nonzero(np.asarray(mask, dtype=bool) & lesion) / voxels
                    >= DETECTION_FRACTION
                )
                for key, mask in candidates.items()
            }
        )
    candidate_components = {}
    for key, mask in candidates.items():
        labels, count = ndimage.label(np.asarray(mask, dtype=bool), structure=structure)
        sizes = np.bincount(labels.ravel(), minlength=count + 1)
        overlap = np.bincount(labels[gt.astype(bool)], minlength=count + 1)
        eligible = np.flatnonzero(sizes * voxel_mm3 >= HIGH_VALUE_MM3)
        eligible = eligible[eligible > 0]
        total = int(eligible.size)
        supported = int(
            np.count_nonzero(
                overlap[eligible] / sizes[eligible] >= DETECTION_FRACTION
            )
        )
        candidate_components[key] = {"total": total, "gt_supported": supported}
    return {"missed": missed, "candidate_components": candidate_components}


def select_deployment(deployments: list[dict]) -> tuple[dict, dict, bool]:
    if not deployments:
        raise ValueError("no STU-Net deployments to select")
    ordered = sorted(
        deployments,
        key=lambda row: (row["Lesion_DSC"], row["Lesion_NSD"]),
        reverse=True,
    )
    qualified = [
        row
        for row in ordered
        if row["Lesion_DSC"] >= DSC_GATE and row["Lesion_NSD"] >= NSD_FLOOR
    ]
    return (qualified[0] if qualified else ordered[0]), ordered[0], bool(qualified)


def score_one_case(
    item: tuple[str, str, str, str, str, str | None, bool],
) -> tuple[str, dict[str, tuple[float, float]], dict]:
    """Load and score one case so public screens can use independent CPUs."""

    name, labels_dir, anchor_dir, stunet_dir, freetumor_dir, probability_dir, closed_only = item
    gt_image = nib.load(str(Path(labels_dir) / name))
    gt = np.asanyarray(gt_image.dataobj).astype(bool, copy=False)
    loaded = []
    for directory in map(Path, (anchor_dir, stunet_dir, freetumor_dir)):
        image = nib.load(str(directory / name))
        if image.shape != gt_image.shape or not np.allclose(
            image.affine, gt_image.affine, atol=1e-4
        ):
            raise ValueError(f"geometry mismatch for {name}: {directory}")
        values = np.asanyarray(image.dataobj)
        if not np.isin(np.unique(values), [0, 1]).all():
            raise ValueError(f"nonbinary prediction for {name}: {directory}")
        loaded.append(values.astype(bool, copy=False))
    voxel_mm3 = float(abs(np.linalg.det(gt_image.affine[:3, :3])))
    spacing = gt_image.header.get_zooms()[:3]
    probability = (
        load_compact_probability(
            Path(probability_dir) / probability_name(name), gt_image.shape
        )
        if probability_dir is not None
        else None
    )
    masks = configurations(*loaded, voxel_mm3, probability, closed_only)
    candidate_masks = {"p50": loaded[1]}
    if probability is not None and not closed_only:
        candidate_masks.update(
            {
                f"p{int(round(threshold * 100)):02d}": probability > threshold
                for threshold in PROBABILITY_THRESHOLDS
            }
        )
    elif probability is not None:
        candidate_masks["p70"] = probability > 0.70
    reference = prepare_surface_reference(gt, spacing) if gt.any() else None
    metric_cache: dict[bytes, tuple[float, float]] = {}
    metrics = {}
    for key, mask in masks.items():
        packed = np.packbits(np.asarray(mask, dtype=bool).reshape(-1), bitorder="little")
        digest = hashlib.sha256(packed).digest()
        if digest not in metric_cache:
            metric_cache[digest] = score_case_with_reference(
                gt, mask, spacing, reference
            )
        metrics[key] = metric_cache[digest]
    diagnostic = high_value_diagnostic(
        gt, loaded[0], candidate_masks, voxel_mm3
    )
    diagnostic["metric_configurations"] = len(masks)
    diagnostic["unique_metric_masks"] = len(metric_cache)
    return name, metrics, diagnostic


def evaluate(
    labels_dir: Path,
    anchor_dir: Path,
    stunet_dir: Path,
    freetumor_dir: Path,
    workers: int = 1,
    stunet_probabilities_dir: Path | None = None,
    case_cache_dir: Path | None = None,
    closed_only: bool = False,
) -> dict:
    names = sorted(path.name for path in labels_dir.glob("*.nii.gz"))
    expected = set(names)
    if not names:
        raise FileNotFoundError(f"no labels in {labels_dir}")
    for role, directory in (
        ("anchor", anchor_dir),
        ("stunet", stunet_dir),
        ("freetumor", freetumor_dir),
    ):
        observed = {path.name for path in directory.glob("*.nii.gz")}
        if observed != expected:
            raise ValueError(
                f"{role} inventory mismatch: missing={sorted(expected-observed)} "
                f"extra={sorted(observed-expected)}"
            )
    if stunet_probabilities_dir is not None:
        observed = {path.name for path in stunet_probabilities_dir.glob("*.prob16.npz")}
        expected_probabilities = {probability_name(name) for name in names}
        if observed != expected_probabilities:
            raise ValueError(
                "STU-Net probability inventory mismatch: "
                f"missing={sorted(expected_probabilities-observed)} "
                f"extra={sorted(observed-expected_probabilities)}"
            )

    if workers < 1:
        raise ValueError("workers must be positive")
    items = [
        (
            name,
            str(labels_dir),
            str(anchor_dir),
            str(stunet_dir),
            str(freetumor_dir),
            str(stunet_probabilities_dir) if stunet_probabilities_dir else None,
            closed_only,
        )
        for name in names
    ]
    if case_cache_dir is not None:
        case_cache_dir.mkdir(parents=True, exist_ok=True)
    signatures = {
        item[0]: case_source_signature(item) for item in items
    } if case_cache_dir is not None else {}
    completed: dict[str, tuple[str, dict[str, tuple[float, float]], dict]] = {}
    pending = []
    cache_reused = 0
    cache_written = 0
    for item in items:
        name = item[0]
        cached = (
            load_case_cache(
                case_cache_path(case_cache_dir, name), name, signatures[name]
            )
            if case_cache_dir is not None
            else None
        )
        if cached is None:
            pending.append(item)
        else:
            completed[name] = cached
            cache_reused += 1
            print(
                f"public metric progress {len(completed)}/{len(items)}: {name} (cache)",
                flush=True,
            )

    def save_result(result: tuple[str, dict[str, tuple[float, float]], dict]) -> None:
        nonlocal cache_written
        name = result[0]
        completed[name] = result
        if case_cache_dir is not None:
            write_case_cache(
                case_cache_path(case_cache_dir, name), result, signatures[name]
            )
            cache_written += 1
        print(
            f"public metric progress {len(completed)}/{len(items)}: {name}",
            flush=True,
        )

    if workers == 1:
        for result in map(score_one_case, pending):
            save_result(result)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(score_one_case, item): item[0] for item in pending}
            for future in as_completed(futures):
                save_result(future.result())

    if set(completed) != expected:
        raise RuntimeError(
            f"incomplete case metrics: missing={sorted(expected-set(completed))} "
            f"extra={sorted(set(completed)-expected)}"
        )
    rows: dict[str, list[tuple[float, float]]] = {}
    diagnostics = []
    configuration_keys = None
    for case in names:
        _, values, diagnostic = completed[case]
        keys = set(values)
        if configuration_keys is None:
            configuration_keys = keys
        elif keys != configuration_keys:
            raise ValueError(
                f"configuration mismatch for {case}: "
                f"missing={sorted(configuration_keys-keys)} "
                f"extra={sorted(keys-configuration_keys)}"
            )
        for key, metric in values.items():
            rows.setdefault(key, []).append(metric)
        diagnostics.append({"case": case, **diagnostic})

    metrics = {}
    for key, values in rows.items():
        array = np.asarray(values, dtype=float)
        metrics[key] = {
            "Lesion_DSC": float(array[:, 0].mean()),
            "Lesion_NSD": float(array[:, 1].mean()),
        }
    anchor = metrics["anchor"]
    deployments = []
    for key, value in metrics.items():
        if key == "anchor":
            continue
        deployments.append(
            {
                "configuration": key,
                **value,
                "DSC_gain": value["Lesion_DSC"] - anchor["Lesion_DSC"],
                "NSD_gain": value["Lesion_NSD"] - anchor["Lesion_NSD"],
            }
        )
    deployments.sort(
        key=lambda row: (row["Lesion_DSC"], row["Lesion_NSD"]), reverse=True
    )
    best, best_unconstrained, gate = select_deployment(deployments)
    diagnostic_keys = sorted(diagnostics[0]["candidate_components"])
    missed_count = sum(len(item["missed"]) for item in diagnostics)
    recovered = {
        key: sum(
            int(miss[key])
            for item in diagnostics
            for miss in item["missed"]
        )
        for key in diagnostic_keys
    }
    component_quality = {}
    for key in diagnostic_keys:
        total = sum(item["candidate_components"][key]["total"] for item in diagnostics)
        supported = sum(
            item["candidate_components"][key]["gt_supported"]
            for item in diagnostics
        )
        component_quality[key] = {
            "total_ge3000mm3": total,
            "gt_supported_ge10pct": supported,
            "support_precision": supported / total if total else None,
        }
    return {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "case_count": len(names),
        "case_metric_cache": {
            "enabled": case_cache_dir is not None,
            "directory": str(case_cache_dir.resolve()) if case_cache_dir else None,
            "reused": cache_reused,
            "written": cache_written,
        },
        "selection_policy": (
            "closed_STUNet_standalone_and_fixed_p70_anchor_confirmation"
            if closed_only
            else "predeclared_STUNet_standalone_GF_union_or_independently_confirmed_anchor_components"
        ),
        "closed_only": closed_only,
        "connectivity": 6,
        "anchor_confirmation_connectivity": 26,
        "minimum_component_mm3": list(MINIMUM_COMPONENT_MM3),
        "freetumor_support_fractions": list(SUPPORT_FRACTIONS),
        "probability_thresholds": (
            list(PROBABILITY_THRESHOLDS) if stunet_probabilities_dir else []
        ),
        "anchor_confirmation_probability_thresholds": (
            list(ANCHOR_CONFIRMATION_PROBABILITY_THRESHOLDS)
            if stunet_probabilities_dir
            else []
        ),
        "metrics": metrics,
        "deployments": deployments,
        "best_deployment": best,
        "best_unconstrained_deployment": best_unconstrained,
        "fixed_public_DSC_gate": DSC_GATE,
        "fixed_public_NSD_floor": NSD_FLOOR,
        "public_gate": gate,
        "high_value_public_diagnostic": {
            "minimum_lesion_mm3": HIGH_VALUE_MM3,
            "detection_fraction": DETECTION_FRACTION,
            "anchor_missed_lesions": missed_count,
            "recovered_by_threshold": recovered,
            "candidate_component_quality": component_quality,
            "selection_authority": False,
        },
        "decision": "QUALIFY_SUBMISSION_PREPARATION" if gate else "KEEP_TOUCHCOMP",
        "role": "fixed_public_ensemble_selection_only",
        "public_labels_read": True,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--anchor", required=True, type=Path)
    parser.add_argument("--stunet", required=True, type=Path)
    parser.add_argument("--freetumor", required=True, type=Path)
    parser.add_argument("--stunet-probabilities", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--case-cache-dir", type=Path)
    parser.add_argument(
        "--closed-only",
        action="store_true",
        help="score only anchor, standalone STU-Net, and fixed p70 confirmation",
    )
    args = parser.parse_args()
    case_cache_dir = args.case_cache_dir or args.output.with_name(
        f"{args.output.stem}_case_metrics"
    )
    payload = evaluate(
        args.labels,
        args.anchor,
        args.stunet,
        args.freetumor,
        args.workers,
        args.stunet_probabilities,
        case_cache_dir,
        args.closed_only,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
