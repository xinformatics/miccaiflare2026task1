#!/usr/bin/env python3
"""Build a dense internal-Dice curve over a retained checkpoint trajectory.

The runner is intentionally not started by the training controller.  Invoke it
after higher-priority training/public-gate jobs finish.  It fails closed unless
the complete checkpoint range exists and every requested GPU is idle, then
runs one independent checkpoint evaluation per GPU in resumable waves.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EVALUATOR = ROOT / "evaluate_internal_checkpoint.py"
DEFAULT_RUN = ROOT / "runs/gf_focus_pu_long30_e283_to313_20260825"
DEFAULT_OUTPUT = (
    ROOT / "public_sweeps/gf_focus_pu_long30_internal_grid_20260825"
)
INLINE_VALIDATION_PATTERN = re.compile(
    r"Final validation\s+(?P<epoch>\d+)/(?P<last_epoch>\d+)\s+"
    r"dice\s+(?P<dice>[0-9.eE+-]+)\s+time\s+"
    r"(?P<elapsed>[0-9.eE+-]+)s"
)


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def gpu_memory_mib(gpu: int) -> int:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip())


def checkpoint_path(run_dir: Path, epoch: int) -> Path:
    return run_dir / f"checkpoint_epoch_{epoch:04d}.pt"


def result_path(output: Path, epoch: int) -> Path:
    return output / f"checkpoint_epoch_{epoch:04d}.internal.json"


def parity_result_path(output: Path, epoch: int) -> Path:
    return output / f"checkpoint_epoch_{epoch:04d}.parity.internal.json"


def load_complete_result(path: Path, checkpoint: Path) -> dict | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETE":
        raise ValueError(f"existing result is not COMPLETE: {path}")
    recorded = payload.get("checkpoint", {})
    if Path(recorded.get("path", "")).resolve() != checkpoint.resolve():
        raise ValueError(f"checkpoint path mismatch in existing result: {path}")
    if recorded.get("sha256") != sha256_file(checkpoint):
        raise ValueError(f"checkpoint hash mismatch in existing result: {path}")
    if payload.get("public_read") or payload.get("hidden_read"):
        raise ValueError(f"internal audit unexpectedly read challenge data: {path}")
    source = payload.get("evaluation_source", "standalone_single_gpu_validation")
    if source not in {
        "inline_distributed_training_validation",
        "standalone_single_gpu_validation",
    }:
        raise ValueError(f"unknown internal evaluation source in {path}: {source}")
    if source == "standalone_single_gpu_validation":
        evaluator = payload.get("evaluator", {})
        if (
            Path(str(evaluator.get("path", ""))).resolve() != EVALUATOR.resolve()
            or evaluator.get("sha256") != sha256_file(EVALUATOR)
        ):
            raise ValueError(f"standalone evaluator identity mismatch: {path}")
    return payload


def parse_inline_validations(train_log: Path) -> dict[int, dict]:
    """Return checkpoint-indexed inline validation results from a stable log."""
    if not train_log.is_file():
        raise FileNotFoundError(f"missing training log: {train_log}")
    results = {}
    for match in INLINE_VALIDATION_PATTERN.finditer(
        train_log.read_text(encoding="utf-8")
    ):
        checkpoint_epoch = int(match.group("epoch")) + 1
        current = {
            "dice": float(match.group("dice")),
            "elapsed_seconds": float(match.group("elapsed")),
            "logged_epoch": int(match.group("epoch")),
            "logged_last_epoch": int(match.group("last_epoch")),
        }
        previous = results.get(checkpoint_epoch)
        if previous is not None and previous != current:
            raise ValueError(
                f"conflicting inline validation records for checkpoint {checkpoint_epoch}"
            )
        results[checkpoint_epoch] = current
    return results


def import_inline_results(
    output: Path,
    inventory: list[tuple[int, Path]],
    train_log: Path,
    dataset_cases: int = 90,
    world_size: int = 2,
) -> list[int]:
    if dataset_cases < 1 or world_size < 1 or dataset_cases % world_size:
        raise ValueError("inline case count must be positive and divisible by world size")
    parsed = parse_inline_validations(train_log)
    log_identity = {
        "path": str(train_log.resolve()),
        "sha256": sha256_file(train_log),
    }
    imported = []
    for epoch, checkpoint in inventory:
        record = parsed.get(epoch)
        if record is None:
            continue
        audit = result_path(output, epoch)
        if audit.exists():
            load_complete_result(audit, checkpoint)
            continue
        payload = {
            "schema_version": 1,
            "created_at": dt.datetime.now().astimezone().isoformat(),
            "status": "COMPLETE",
            "checkpoint": {
                "path": str(checkpoint.resolve()),
                "sha256": sha256_file(checkpoint),
                "bytes": checkpoint.stat().st_size,
                "embedded_completed_epoch": epoch - 1,
            },
            "internal_validation": {
                "dataset_cases": dataset_cases,
                "distributed_steps": dataset_cases // world_size,
                "world_size": world_size,
                "dice": record["dice"],
                "elapsed_seconds": record["elapsed_seconds"],
            },
            "evaluation_source": "inline_distributed_training_validation",
            "training_log": log_identity,
            "role": "internal_diagnostic_only_not_checkpoint_promotion",
            "public_read": False,
            "hidden_read": False,
            "submission_created": False,
            "uploaded": False,
        }
        atomic_json(audit, payload)
        imported.append(epoch)
    return imported


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a complete GF-Screen checkpoint range on internal validation"
    )
    parser.add_argument("--run-dir", default=DEFAULT_RUN, type=Path)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument("--start-epoch", default=284, type=int)
    parser.add_argument("--end-epoch", default=313, type=int)
    parser.add_argument("--gpus", default=(0, 1), nargs="+", type=int)
    parser.add_argument("--workers", default=2, type=int)
    parser.add_argument("--idle-memory-mib", default=4096, type=int)
    parser.add_argument(
        "--parity-epoch",
        default=295,
        type=int,
        help="recompute this imported inline point standalone to prove source parity",
    )
    parser.add_argument("--parity-tolerance", default=1e-6, type=float)
    parser.add_argument(
        "--no-import-inline",
        action="store_true",
        help="recompute scheduled validation checkpoints instead of importing train.log results",
    )
    return parser


def validate_cli(args: argparse.Namespace) -> list[tuple[int, Path]]:
    if args.start_epoch < 0 or args.end_epoch < args.start_epoch:
        raise ValueError("invalid checkpoint epoch range")
    if not args.gpus or len(set(args.gpus)) != len(args.gpus):
        raise ValueError("--gpus must contain unique device indices")
    if min(args.gpus) < 0 or args.workers < 0 or args.idle_memory_mib < 0:
        raise ValueError("GPU indices, workers, and idle-memory threshold must be non-negative")
    if not args.start_epoch <= args.parity_epoch <= args.end_epoch:
        raise ValueError("--parity-epoch must lie inside the checkpoint range")
    if args.parity_tolerance < 0:
        raise ValueError("--parity-tolerance must be non-negative")
    if not EVALUATOR.is_file():
        raise FileNotFoundError(f"missing evaluator: {EVALUATOR}")
    inventory = [
        (epoch, checkpoint_path(args.run_dir, epoch))
        for epoch in range(args.start_epoch, args.end_epoch + 1)
    ]
    missing = [str(path) for _, path in inventory if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "complete checkpoint range is required before dense evaluation; missing: "
            + ", ".join(missing)
        )
    return inventory


def evaluate_one(
    epoch: int,
    checkpoint: Path,
    output: Path,
    gpu: int,
    workers: int,
    audit: Path | None = None,
    log: Path | None = None,
) -> tuple[subprocess.Popen, object]:
    audit = audit or result_path(output, epoch)
    log = log or output / f"checkpoint_epoch_{epoch:04d}.log"
    environment = dict(os.environ)
    environment["MPLCONFIGDIR"] = str(output / ".matplotlib")
    command = [
        sys.executable,
        str(EVALUATOR),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(audit),
        "--gpu",
        str(gpu),
        "--workers",
        str(workers),
    ]
    handle = log.open("a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    return process, handle


def compare_source_parity(
    inline_payload: dict,
    standalone_payload: dict,
    epoch: int,
    tolerance: float,
) -> dict:
    inline_source = inline_payload.get("evaluation_source")
    if inline_source != "inline_distributed_training_validation":
        raise ValueError(
            f"checkpoint {epoch} parity anchor is not an imported inline result"
        )
    standalone_source = standalone_payload.get(
        "evaluation_source", "standalone_single_gpu_validation"
    )
    if standalone_source != "standalone_single_gpu_validation":
        raise ValueError(
            f"checkpoint {epoch} parity candidate is not standalone single-GPU"
        )
    inline_dice = float(inline_payload["internal_validation"]["dice"])
    standalone_dice = float(standalone_payload["internal_validation"]["dice"])
    absolute_difference = abs(inline_dice - standalone_dice)
    if absolute_difference > tolerance:
        raise ValueError(
            f"checkpoint {epoch} inline/standalone Dice mismatch: "
            f"{absolute_difference} > {tolerance}"
        )
    return {
        "status": "PASS",
        "epoch": epoch,
        "inline_dice": inline_dice,
        "standalone_dice": standalone_dice,
        "absolute_difference": absolute_difference,
        "tolerance": tolerance,
    }


RECALL_SUMMARY_FIELDS = (
    "annotated_voxel_recall",
    "annotated_component_recall",
    "annotated_component_recall_lt100",
    "annotated_component_recall_100to500",
    "annotated_component_recall_500to1000",
    "annotated_component_recall_1000to3000",
    "annotated_component_recall_ge3000",
)
HIGH_VALUE_COMPONENT_RECALL_FIELD = "annotated_component_recall_ge3000"
RECALL_SHORTLIST_DICE_GUARD = 0.01
RECALL_SHORTLIST_MAX_CANDIDATES = 3


def write_summary(
    output: Path,
    inventory: list[tuple[int, Path]],
    recall_overrides: dict[int, dict] | None = None,
) -> list[dict]:
    """Write checkpoint metrics, retaining recall from standalone parity runs.

    Inline validations produced by the already-running trainer predate the
    annotated-recall instrumentation.  The independently recomputed parity
    checkpoint is authoritative for those additional metrics, so callers may
    supply it as an epoch-keyed override without altering the recorded Dice.
    """
    recall_overrides = recall_overrides or {}
    rows = []
    for epoch, checkpoint in inventory:
        payload = load_complete_result(result_path(output, epoch), checkpoint)
        if payload is None:
            raise RuntimeError(f"missing completed result for epoch {epoch}")
        internal = payload["internal_validation"]
        recall = recall_overrides.get(epoch, internal.get("annotated_recall", {}))
        row = {
            "epoch": epoch,
            "dice": float(internal["dice"]),
            "cases": int(internal["dataset_cases"]),
            "elapsed_seconds": float(internal["elapsed_seconds"]),
            "checkpoint_sha256": payload["checkpoint"]["sha256"],
            "evaluation_source": payload.get(
                "evaluation_source", "standalone_single_gpu_validation"
            ),
            "recall_evaluation_source": (
                "standalone_parity_override"
                if epoch in recall_overrides
                else (
                    payload.get(
                        "evaluation_source", "standalone_single_gpu_validation"
                    )
                    if recall
                    else "unavailable_pre_instrumentation"
                )
            ),
            "audit": str(result_path(output, epoch).resolve()),
        }
        for field in RECALL_SUMMARY_FIELDS:
            value = recall.get(field)
            row[field] = None if value is None else float(value)
        rows.append(row)
    temporary = output / f".summary.tsv.{os.getpid()}.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output / "summary.tsv")
    return rows


def build_curve_analysis(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("internal learning curve requires at least one row")
    points = []
    for row in rows:
        point = {
            "epoch": int(row["epoch"]),
            "dice": float(row["dice"]),
            "evaluation_source": str(row["evaluation_source"]),
            "recall_evaluation_source": str(
                row.get("recall_evaluation_source", "unavailable")
            ),
            "checkpoint_sha256": str(row.get("checkpoint_sha256", "")),
            "audit": str(row.get("audit", "")),
        }
        for field in RECALL_SUMMARY_FIELDS:
            value = row.get(field)
            point[field] = None if value is None else float(value)
        points.append(point)
    deltas = [
        {
            "from_epoch": left["epoch"],
            "to_epoch": right["epoch"],
            "dice_change": right["dice"] - left["dice"],
        }
        for left, right in zip(points, points[1:])
    ]
    best = max(points, key=lambda point: (point["dice"], point["epoch"]))
    minimum = min(points, key=lambda point: (point["dice"], point["epoch"]))
    largest_gain = max(deltas, key=lambda delta: delta["dice_change"]) if deltas else None
    largest_drop = min(deltas, key=lambda delta: delta["dice_change"]) if deltas else None
    recall_points = [
        point for point in points
        if point["annotated_component_recall"] is not None
        and point["annotated_voxel_recall"] is not None
    ]
    best_component_recall = (
        max(
            recall_points,
            key=lambda point: (point["annotated_component_recall"], point["epoch"]),
        )
        if recall_points
        else None
    )
    best_voxel_recall = (
        max(
            recall_points,
            key=lambda point: (point["annotated_voxel_recall"], point["epoch"]),
        )
        if recall_points
        else None
    )
    large_recall_points = [
        point
        for point in recall_points
        if point.get(HIGH_VALUE_COMPONENT_RECALL_FIELD) is not None
    ]
    best_large_component_recall = (
        max(
            large_recall_points,
            key=lambda point: (
                point[HIGH_VALUE_COMPONENT_RECALL_FIELD],
                point["dice"],
                point["epoch"],
            ),
        )
        if large_recall_points
        else None
    )
    return {
        "schema_version": 1,
        "checkpoint_count": len(points),
        "checkpoint_range": [points[0]["epoch"], points[-1]["epoch"]],
        "points": points,
        "best": best,
        "minimum": minimum,
        "first_to_last_dice_change": points[-1]["dice"] - points[0]["dice"],
        "largest_adjacent_gain": largest_gain,
        "largest_adjacent_drop": largest_drop,
        "recall_available_checkpoint_count": len(recall_points),
        "best_annotated_component_recall": best_component_recall,
        "best_annotated_large_component_recall_ge3000": (
            best_large_component_recall
        ),
        "best_annotated_voxel_recall": best_voxel_recall,
        "evaluation_source_counts": {
            source: sum(point["evaluation_source"] == source for point in points)
            for source in sorted({point["evaluation_source"] for point in points})
        },
        "role": "internal_diagnostic_only_not_checkpoint_promotion",
    }


def build_fixed_public_shortlist(
    analysis: dict,
    dice_guard_delta: float = RECALL_SHORTLIST_DICE_GUARD,
    maximum_candidates: int = RECALL_SHORTLIST_MAX_CANDIDATES,
) -> dict:
    """Choose complementary internal candidates without promoting any of them.

    Partial-label-safe recall is useful for deciding which checkpoints deserve
    complete-label public evaluation, but it is not a deployment metric. The
    measured fixed-public oracle assigns nearly all recoverable macro DSC to
    missed lesions of at least 3000 mm3, so unweighted lesion-count and
    sub-1000-mm3 recall deliberately do not nominate checkpoints. A tight Dice
    guard prevents a high-recall collapsed mask from entering the screen; the
    fixed public 50 remains the sole promotion authority.
    """
    if not 0.0 <= dice_guard_delta <= 1.0:
        raise ValueError("Dice guard delta must lie in [0, 1]")
    if maximum_candidates < 1:
        raise ValueError("maximum candidates must be positive")
    points = list(analysis.get("points", ()))
    if not points:
        raise ValueError("shortlist requires at least one learning-curve point")
    best_dice = max(float(point["dice"]) for point in points)
    dice_floor = best_dice - dice_guard_delta
    recall_eligible = [
        point
        for point in points
        if float(point["dice"]) >= dice_floor
        and point.get("annotated_component_recall") is not None
        and point.get("annotated_voxel_recall") is not None
    ]

    selected: list[dict] = []

    def add(role: str, point: dict, selection_metric: str, value: float) -> None:
        for candidate in selected:
            if int(candidate["epoch"]) == int(point["epoch"]):
                candidate["selection_roles"].append(role)
                candidate["selection_metrics"][selection_metric] = float(value)
                return
        if len(selected) >= maximum_candidates:
            return
        selected.append(
            {
                "epoch": int(point["epoch"]),
                "dice": float(point["dice"]),
                "annotated_component_recall": point.get(
                    "annotated_component_recall"
                ),
                "annotated_voxel_recall": point.get("annotated_voxel_recall"),
                "annotated_component_recall_ge3000": point.get(
                    HIGH_VALUE_COMPONENT_RECALL_FIELD
                ),
                "checkpoint_sha256": point.get("checkpoint_sha256"),
                "internal_audit": point.get("audit"),
                "selection_roles": [role],
                "selection_metrics": {selection_metric: float(value)},
            }
        )

    best_dice_point = max(
        points, key=lambda point: (float(point["dice"]), int(point["epoch"]))
    )
    add("best_internal_dice", best_dice_point, "dice", best_dice)
    if recall_eligible:
        large_eligible = [
            point
            for point in recall_eligible
            if point.get(HIGH_VALUE_COMPONENT_RECALL_FIELD) is not None
        ]
        if large_eligible:
            large = max(
                large_eligible,
                key=lambda point: (
                    float(point[HIGH_VALUE_COMPONENT_RECALL_FIELD]),
                    float(point["dice"]),
                    int(point["epoch"]),
                ),
            )
            add(
                "best_large_component_recall_ge3000_within_dice_guard",
                large,
                HIGH_VALUE_COMPONENT_RECALL_FIELD,
                float(large[HIGH_VALUE_COMPONENT_RECALL_FIELD]),
            )
        voxel = max(
            recall_eligible,
            key=lambda point: (
                float(point["annotated_voxel_recall"]),
                float(point["dice"]),
                int(point["epoch"]),
            ),
        )
        add(
            "best_voxel_recall_within_dice_guard",
            voxel,
            "annotated_voxel_recall",
            float(voxel["annotated_voxel_recall"]),
        )
    return {
        "schema_version": 1,
        "status": "COMPLETE",
        "selection_basis": (
            "locked_internal_partial_label_safe_high_value_diagnostics"
        ),
        "selection_objective": (
            "case_balanced_dice_plus_ge3000mm3_and_voxel_recall"
        ),
        "large_component_minimum_mm3": 3000,
        "best_internal_dice": best_dice,
        "internal_dice_floor": dice_floor,
        "dice_guard_delta": dice_guard_delta,
        "maximum_candidates": maximum_candidates,
        "candidate_count": len(selected),
        "candidates": selected,
        "promotion_authority": "fixed_public_50_only",
        "fixed_public_DSC_gate": 0.6154316379500749,
        "fixed_public_NSD_floor": 0.5104225107769019,
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }


def write_curve_svg(output: Path, analysis: dict) -> Path:
    """Write a dependency-free SVG so the service never needs matplotlib."""
    points = analysis["points"]
    width, height = 960, 540
    left, right, top, bottom = 88, 32, 58, 74
    plot_width = width - left - right
    plot_height = height - top - bottom
    epochs = [point["epoch"] for point in points]
    values = [point["dice"] for point in points]
    x_min, x_max = min(epochs), max(epochs)
    raw_min, raw_max = min(values), max(values)
    padding = max(0.005, (raw_max - raw_min) * 0.15)
    y_min = max(0.0, raw_min - padding)
    y_max = min(1.0, raw_max + padding)
    if y_max <= y_min:
        y_min, y_max = max(0.0, raw_min - 0.01), min(1.0, raw_max + 0.01)

    def x(epoch: int) -> float:
        return left + (epoch - x_min) / max(1, x_max - x_min) * plot_width

    def y(value: float) -> float:
        return top + (y_max - value) / max(1e-12, y_max - y_min) * plot_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="480" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">Locked internal Dice by checkpoint</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#222"/>',
    ]
    for index in range(6):
        value = y_min + (y_max - y_min) * index / 5
        coordinate = y(value)
        svg.extend([
            f'<line x1="{left}" y1="{coordinate:.2f}" x2="{left + plot_width}" y2="{coordinate:.2f}" stroke="#e5e7eb"/>',
            f'<text x="{left - 10}" y="{coordinate + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.3f}</text>',
        ])
    epoch_ticks = sorted({epochs[0], epochs[-1], *epochs[::5]})
    for epoch in epoch_ticks:
        coordinate = x(epoch)
        svg.extend([
            f'<line x1="{coordinate:.2f}" y1="{top + plot_height}" x2="{coordinate:.2f}" y2="{top + plot_height + 5}" stroke="#222"/>',
            f'<text x="{coordinate:.2f}" y="{top + plot_height + 23}" text-anchor="middle" font-family="sans-serif" font-size="12">{epoch}</text>',
        ])
    polyline = " ".join(
        f'{x(point["epoch"]):.2f},{y(point["dice"]):.2f}' for point in points
    )
    svg.append(
        f'<polyline points="{polyline}" fill="none" stroke="#2563eb" stroke-width="2.5"/>'
    )
    for point in points:
        inline = point["evaluation_source"].startswith("inline_")
        color = "#f59e0b" if inline else "#2563eb"
        svg.append(
            f'<circle cx="{x(point["epoch"]):.2f}" cy="{y(point["dice"]):.2f}" r="4" fill="{color}"><title>epoch {point["epoch"]}: Dice {point["dice"]:.8f} ({point["evaluation_source"]})</title></circle>'
        )
    best = analysis["best"]
    svg.extend([
        f'<text x="{x(best["epoch"]):.2f}" y="{max(46, y(best["dice"]) - 12):.2f}" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="bold">best e{best["epoch"]}: {best["dice"]:.4f}</text>',
        f'<text x="{left + plot_width / 2:.2f}" y="{height - 20}" text-anchor="middle" font-family="sans-serif" font-size="14">Checkpoint epoch</text>',
        f'<text x="20" y="{top + plot_height / 2:.2f}" text-anchor="middle" transform="rotate(-90 20 {top + plot_height / 2:.2f})" font-family="sans-serif" font-size="14">Internal Dice</text>',
        f'<circle cx="{width - 260}" cy="30" r="4" fill="#f59e0b"/><text x="{width - 250}" y="34" font-family="sans-serif" font-size="12">inline DDP</text>',
        f'<circle cx="{width - 150}" cy="30" r="4" fill="#2563eb"/><text x="{width - 140}" y="34" font-family="sans-serif" font-size="12">deferred single-GPU</text>',
        '</svg>\n',
    ])
    path = output / "internal_dice_curve.svg"
    atomic_text(path, "\n".join(svg))
    return path


def main() -> None:
    args = build_cli().parse_args()
    args.run_dir = args.run_dir.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    inventory = validate_cli(args)

    lock_path = args.output / "RUN.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        imported_epochs = []
        if not args.no_import_inline:
            imported_epochs = import_inline_results(
                args.output, inventory, args.run_dir / "train.log"
            )

        pending = []
        for epoch, checkpoint in inventory:
            if load_complete_result(result_path(args.output, epoch), checkpoint) is None:
                pending.append((epoch, checkpoint))

        while pending:
            wave = pending[: len(args.gpus)]
            pending = pending[len(args.gpus) :]
            for gpu in args.gpus[: len(wave)]:
                used = gpu_memory_mib(gpu)
                if used > args.idle_memory_mib:
                    raise RuntimeError(
                        f"GPU {gpu} is busy ({used} MiB > {args.idle_memory_mib} MiB); "
                        "dense internal evaluation must not contend with training/inference"
                    )

            processes = []
            for (epoch, checkpoint), gpu in zip(wave, args.gpus):
                process, handle = evaluate_one(
                    epoch, checkpoint, args.output, gpu, args.workers
                )
                processes.append((epoch, process, handle))
            failures = []
            for epoch, process, handle in processes:
                status = process.wait()
                handle.close()
                if status != 0:
                    failures.append((epoch, status))
            if failures:
                raise RuntimeError(f"checkpoint evaluation failures: {failures}")

        # The final curve mixes imported two-rank inline values with deferred
        # one-GPU values. Recompute one fixed imported point through the
        # standalone path and fail closed unless both reductions agree.
        parity_checkpoint = checkpoint_path(args.run_dir, args.parity_epoch)
        inline_payload = load_complete_result(
            result_path(args.output, args.parity_epoch), parity_checkpoint
        )
        if inline_payload is None:
            raise RuntimeError("missing parity anchor result")
        parity_path = parity_result_path(args.output, args.parity_epoch)
        standalone_payload = load_complete_result(parity_path, parity_checkpoint)
        if standalone_payload is None:
            parity_gpu = args.gpus[0]
            used = gpu_memory_mib(parity_gpu)
            if used > args.idle_memory_mib:
                raise RuntimeError(
                    f"GPU {parity_gpu} is busy ({used} MiB > {args.idle_memory_mib} MiB) "
                    "before source-parity validation"
                )
            process, handle = evaluate_one(
                args.parity_epoch,
                parity_checkpoint,
                args.output,
                parity_gpu,
                args.workers,
                audit=parity_path,
                log=args.output / f"checkpoint_epoch_{args.parity_epoch:04d}.parity.log",
            )
            status = process.wait()
            handle.close()
            if status != 0:
                raise RuntimeError(
                    f"checkpoint {args.parity_epoch} source-parity evaluation failed: {status}"
                )
            standalone_payload = load_complete_result(parity_path, parity_checkpoint)
        assert standalone_payload is not None
        source_parity = compare_source_parity(
            inline_payload,
            standalone_payload,
            args.parity_epoch,
            args.parity_tolerance,
        )
        source_parity["standalone_audit"] = {
            "path": str(parity_path.resolve()),
            "sha256": sha256_file(parity_path),
        }

        parity_recall = standalone_payload["internal_validation"].get(
            "annotated_recall"
        )
        if not parity_recall:
            raise RuntimeError(
                "standalone source-parity result lacks annotated-recall diagnostics"
            )
        rows = write_summary(
            args.output,
            inventory,
            recall_overrides={args.parity_epoch: parity_recall},
        )
        analysis = build_curve_analysis(rows)
        analysis["created_at"] = dt.datetime.now().astimezone().isoformat()
        analysis["summary_sha256"] = sha256_file(args.output / "summary.tsv")
        curve_json = args.output / "learning_curve.json"
        atomic_json(curve_json, analysis)
        curve_svg = write_curve_svg(args.output, analysis)
        shortlist = build_fixed_public_shortlist(analysis)
        shortlist["created_at"] = dt.datetime.now().astimezone().isoformat()
        shortlist["learning_curve"] = {
            "path": str(curve_json.resolve()),
            "sha256": sha256_file(curve_json),
        }
        shortlist_json = args.output / "fixed_public_shortlist.json"
        atomic_json(shortlist_json, shortlist)
        best = analysis["best"]
        terminal = {
            "schema_version": 1,
            "created_at": dt.datetime.now().astimezone().isoformat(),
            "status": "COMPLETE",
            "checkpoint_range": [args.start_epoch, args.end_epoch],
            "evaluated_checkpoints": len(rows),
            "imported_inline_epochs": imported_epochs,
            "internal_cases_per_checkpoint": sorted({row["cases"] for row in rows}),
            "best_epoch": best["epoch"],
            "best_dice": best["dice"],
            "summary": str((args.output / "summary.tsv").resolve()),
            "learning_curve": {
                "path": str(curve_json.resolve()),
                "sha256": sha256_file(curve_json),
            },
            "plot": {
                "path": str(curve_svg.resolve()),
                "sha256": sha256_file(curve_svg),
            },
            "fixed_public_shortlist": {
                "path": str(shortlist_json.resolve()),
                "sha256": sha256_file(shortlist_json),
                "candidate_count": shortlist["candidate_count"],
            },
            "source_parity": source_parity,
            "role": "internal_diagnostic_only_not_checkpoint_promotion",
            "public_read": False,
            "hidden_read": False,
            "submission_created": False,
            "uploaded": False,
        }
        atomic_json(args.output / "TERMINAL.json", terminal)
        print(json.dumps(terminal, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
