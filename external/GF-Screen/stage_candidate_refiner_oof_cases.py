#!/usr/bin/env python3
"""Stage deterministic case-grouped OOF image/proposal/label partitions."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from train_candidate_refiner import grouped_fold_rows, stable_digest


def safe_symlink(source: Path, target: Path) -> None:
    source = source.resolve(strict=True)
    if target.is_symlink() or target.exists():
        if target.resolve(strict=True) != source:
            raise RuntimeError(f"staging collision: {target} -> {target.resolve()} != {source}")
        return
    target.symlink_to(source)


def build_partitions(
    rows: list[dict[str, str]],
    items: list[dict],
    folds: int,
    seed: int,
) -> tuple[dict[int, list[dict]], list[str]]:
    item_by_case = {str(item["name"]): item for item in items}
    if len(item_by_case) != len(items):
        raise ValueError("duplicate manifest case")
    row_cases = {str(row["case"]) for row in rows}
    if not row_cases <= set(item_by_case):
        raise ValueError("index contains a case outside the manifest")
    assignments: dict[str, int] = {}
    for fold in range(folds):
        _, validation = grouped_fold_rows(rows, folds, fold, seed)
        for case in {str(row["case"]) for row in validation}:
            if case in assignments:
                raise ValueError(f"case assigned to multiple folds: {case}")
            assignments[case] = fold
    if set(assignments) != row_cases:
        raise ValueError("row-bearing cases do not have exact OOF coverage")
    zero_patch_cases = sorted(set(item_by_case) - row_cases)
    for case in zero_patch_cases:
        item = item_by_case[case]
        digest = stable_digest(f"{item['cohort']}:{case}:zero-patch", seed)
        assignments[case] = int(digest, 16) % folds
    partitions = {
        fold: sorted(
            (item_by_case[case] for case, assigned in assignments.items() if assigned == fold),
            key=lambda item: str(item["name"]),
        )
        for fold in range(folds)
    }
    flattened = [str(item["name"]) for fold in range(folds) for item in partitions[fold]]
    if len(flattened) != len(items) or set(flattened) != set(item_by_case):
        raise ValueError("partition coverage mismatch")
    return partitions, zero_patch_cases


def parse_proposal_sources(
    legacy: Path | None, values: list[str]
) -> tuple[dict[str, Path], bool]:
    """Parse either the historical single source or named proposal sources."""

    if legacy is not None and values:
        raise ValueError("use either --proposals or repeatable --proposal, not both")
    if legacy is not None:
        return {"proposal": legacy.resolve(strict=True)}, True
    sources: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"proposal must be NAME=DIRECTORY: {value}")
        name, raw_path = value.split("=", 1)
        if (
            not re.fullmatch(r"[A-Za-z0-9_.-]+", name or "")
            or name in (".", "..")
            or name in sources
        ):
            raise ValueError(f"invalid or duplicate proposal source name: {name!r}")
        sources[name] = Path(raw_path).resolve(strict=True)
    if not sources:
        raise ValueError("at least one proposal source is required")
    return sources, False


def stage_partitions(
    rows: list[dict[str, str]],
    items: list[dict],
    proposal_sources: dict[str, Path],
    output: Path,
    folds: int,
    seed: int,
    legacy_layout: bool,
) -> dict:
    partitions, zero_patch_cases = build_partitions(rows, items, folds, seed)
    expected_prediction_names = {str(item["prediction_name"]) for item in items}
    if len(expected_prediction_names) != len(items):
        raise ValueError("duplicate manifest prediction name")
    for name, directory in proposal_sources.items():
        observed = {path.name for path in directory.glob("*.nii.gz")}
        if observed != expected_prediction_names:
            raise ValueError(
                f"proposal coverage mismatch for {name}: "
                f"expected={len(expected_prediction_names)} observed={len(observed)}"
            )

    def proposal_directory(root: Path, name: str) -> Path:
        return root / "proposals" if legacy_layout else root / "proposals" / name

    all_root = output / "all"
    all_images = all_root / "images"
    all_labels = all_root / "labels"
    all_proposals = {
        name: proposal_directory(all_root, name) for name in proposal_sources
    }
    for directory in (all_images, all_labels, *all_proposals.values()):
        directory.mkdir(parents=True, exist_ok=True)

    report_folds = []
    for fold, fold_items in partitions.items():
        fold_root = output / f"fold{fold}"
        fold_images = fold_root / "images"
        fold_labels = fold_root / "labels"
        fold_proposals = {
            name: proposal_directory(fold_root, name) for name in proposal_sources
        }
        for directory in (fold_images, fold_labels, *fold_proposals.values()):
            directory.mkdir(parents=True, exist_ok=True)
        for item in fold_items:
            case = str(item["name"])
            prediction_name = str(item["prediction_name"])
            safe_symlink(
                Path(str(item["image"])), fold_images / f"{case}_0000.nii.gz"
            )
            safe_symlink(Path(str(item["label"])), fold_labels / f"{case}.nii.gz")
            safe_symlink(
                Path(str(item["image"])), all_images / f"{case}_0000.nii.gz"
            )
            safe_symlink(Path(str(item["label"])), all_labels / f"{case}.nii.gz")
            for name, source in proposal_sources.items():
                safe_symlink(
                    source / prediction_name,
                    fold_proposals[name] / f"{case}.nii.gz",
                )
                safe_symlink(
                    source / prediction_name,
                    all_proposals[name] / f"{case}.nii.gz",
                )
        report_folds.append(
            {
                "fold": fold,
                "cases": len(fold_items),
                "row_bearing_cases": sum(
                    str(item["name"]) not in zero_patch_cases for item in fold_items
                ),
                "zero_patch_cases": [
                    str(item["name"])
                    for item in fold_items
                    if str(item["name"]) in zero_patch_cases
                ],
            }
        )
    report = {
        "selection_basis": "same grouped-fold function and seed as training",
        "cases": len(items),
        "row_bearing_cases": len({row["case"] for row in rows}),
        "zero_patch_cases": zero_patch_cases,
        "proposal_layout": "legacy_single" if legacy_layout else "named_sources",
        "proposal_sources": {
            name: str(path.resolve()) for name, path in proposal_sources.items()
        },
        "exact_case_coverage_once": True,
        "exact_proposal_coverage": True,
        "folds": report_folds,
    }
    if legacy_layout:
        report["proposals"] = str(next(iter(proposal_sources.values())).resolve())
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--proposals",
        type=Path,
        help="historical single proposal directory (preserves the old layout)",
    )
    parser.add_argument(
        "--proposal",
        action="append",
        default=[],
        metavar="NAME=DIRECTORY",
        help="repeat to stage named proposal sources under proposals/NAME",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("folds must be at least two")
    with args.index.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    items = json.loads(args.manifest.read_text(encoding="utf-8"))["items"]
    if not rows or not items or any(item.get("complete_label") is not True for item in items):
        raise ValueError("empty data or manifest contains a non-complete label")
    proposal_sources, legacy_layout = parse_proposal_sources(
        args.proposals, args.proposal
    )
    report = stage_partitions(
        rows,
        items,
        proposal_sources,
        args.output,
        args.folds,
        args.seed,
        legacy_layout,
    )
    report["manifest"] = str(args.manifest.resolve())
    report["index"] = str(args.index.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "STAGING_COMPLETE.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
