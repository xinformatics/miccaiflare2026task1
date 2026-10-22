#!/usr/bin/env python3
"""Populate and verify deterministic MONAI cache entries for pseudo positives."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import shutil
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from torch.utils.data import DataLoader, Dataset

from utils.data_utils import get_abdomen_ds


GIB = 1024**3
PROJECT_ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def asset(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def dataset_args(cache_dir: Path, manifest: Path) -> SimpleNamespace:
    """Exactly reproduce the long run's abdominal deterministic prefix."""
    return SimpleNamespace(
        # Services do not have to start in the GF-Screen checkout.  Resolve the
        # base datalist from this module so cache construction is independent
        # of the caller's working directory.
        data_dir=str(PROJECT_ROOT),
        abdomen_json=str(PROJECT_ROOT / "jsons/local/abdomen.json"),
        pseudo_abdomen_json=str(manifest.resolve()),
        pseudo_abdomen_repeat=1,
        use_persistent_dataset=True,
        cache_dir=str(cache_dir.resolve()),
        space_x=1.0,
        space_y=1.0,
        space_z=3.0,
        b_min=0.0,
        b_max=1.0,
        roi_x=96,
        roi_y=96,
        roi_z=64,
        sw_batch_size=2,
        cancer_positive_weight=4,
        cancer_negative_weight=1,
        component_uniform_crop=True,
        component_crop_points_per_lesion=64,
        component_crop_size_power=0.5,
        component_background_points=4096,
        component_crop_cache_dir="",
        component_legacy_cache_dir="",
        enable_scale_intensity_aug=True,
        enable_contrast_aug=True,
        RandScaleIntensityd_prob=0.10,
        RandAdjustContrastd_prob=0.20,
        contrast_gamma_min=0.70,
        contrast_gamma_max=1.50,
        RandFlipd_prob=0.2,
        RandRotate90d_prob=0.2,
        RandShiftIntensityd_prob=0.1,
    )


def pseudo_dataset(cache_dir: Path, manifest: Path):
    dataset, _ = get_abdomen_ds(dataset_args(cache_dir, manifest))
    items = [item for item in dataset.data if item.get("pseudo_label") is True]
    if not items:
        raise ValueError("pseudo-positive cache dataset is empty")
    seen = set()
    unique = []
    for item in items:
        key = dataset.hash_func(item).decode("utf-8") + dataset.transform_hash
        if key in seen:
            raise ValueError(f"duplicate pseudo-positive cache key: {key}")
        seen.add(key)
        unique.append(item)
    dataset.data = unique
    return dataset


def cache_path(dataset, item) -> Path:
    key = dataset.hash_func(item).decode("utf-8") + dataset.transform_hash
    return Path(dataset.cache_dir) / f"{key}.pt"


def cache_inventory(dataset, require_complete: bool = True) -> dict[str, object]:
    records = []
    missing = []
    for item in dataset.data:
        path = cache_path(dataset, item)
        if not path.is_file() or path.stat().st_size < 1:
            missing.append(path.name)
            continue
        records.append((path.name, path.stat().st_size))
    if require_complete and missing:
        raise ValueError(f"missing pseudo cache entries: {len(missing)}")
    rendered = "".join(f"{name}\t{size}\n" for name, size in sorted(records))
    return {
        "members": len(records),
        "bytes": sum(size for _, size in records),
        "name_size_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "missing": len(missing),
    }


class CacheOnlyView(Dataset):
    def __init__(self, persistent_dataset):
        self.dataset = persistent_dataset

    def __len__(self):
        return len(self.dataset.data)

    def __getitem__(self, index):
        item = self.dataset.data[index]
        path = cache_path(self.dataset, item)
        if path.is_file() and path.stat().st_size > 0:
            return {"name": item.get("name", path.stem), "error": "", "cached": True}
        try:
            self.dataset._cachecheck(item)
            if not path.is_file() or path.stat().st_size < 1:
                raise RuntimeError("PersistentDataset did not publish its cache entry")
            return {"name": item.get("name", path.stem), "error": "", "cached": False}
        except Exception as error:  # reported to the parent without tensor IPC
            return {
                "name": item.get("name", path.stem),
                "image": item.get("image", ""),
                "label": item.get("label", ""),
                "error": f"{type(error).__name__}: {error}",
                "cached": False,
            }


def validate_contract(contract: dict, contract_path: Path, manifest_path: Path) -> int:
    if contract.get("status") != "PASS":
        raise ValueError("positive-unlabeled dataset contract did not pass")
    identity = contract.get("pseudo_manifest", {})
    if (
        Path(identity.get("path", "")).resolve() != manifest_path.resolve()
        or identity.get("sha256") != sha256(manifest_path)
    ):
        raise ValueError("pseudo manifest/contract identity mismatch")
    positive_cases = int(contract.get("positive_cases", -1))
    if positive_cases < 1:
        raise ValueError("dataset contract has no pseudo-positive cases")
    if not contract_path.is_file():
        raise ValueError("dataset contract is missing")
    return positive_cases


def verify_marker(
    marker_path: Path,
    dataset,
    manifest_path: Path,
    contract_path: Path,
    base_marker_path: Path,
) -> dict[str, object]:
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "COMPLETE":
        raise ValueError("pseudo cache marker is incomplete")
    for key, path in (
        ("pseudo_manifest", manifest_path),
        ("dataset_contract", contract_path),
        ("base_cache_marker", base_marker_path),
    ):
        identity = marker.get(key, {})
        if (
            Path(identity.get("path", "")).resolve() != path.resolve()
            or identity.get("sha256") != sha256(path)
        ):
            raise ValueError(f"pseudo cache {key} identity mismatch")
    if Path(marker.get("cache_dir", "")).resolve() != Path(dataset.cache_dir).parent.resolve():
        raise ValueError("pseudo cache directory identity mismatch")
    temporary_dir = Path(marker.get("temporary_dir", ""))
    if (
        marker.get("atomic_publication") != "same_filesystem_temporary_then_rename"
        or not temporary_dir.is_dir()
        or int(marker.get("cache_device", -1)) != Path(dataset.cache_dir).stat().st_dev
        or int(marker.get("temporary_device", -2)) != temporary_dir.stat().st_dev
        or temporary_dir.stat().st_dev != Path(dataset.cache_dir).stat().st_dev
    ):
        raise ValueError("pseudo cache atomic-publication proof mismatch")
    live = cache_inventory(dataset)
    recorded = marker.get("cache_inventory", {})
    if any(live[key] != recorded.get(key) for key in ("members", "bytes", "name_size_sha256")):
        raise ValueError("pseudo cache live inventory mismatch")
    if int(marker.get("positive_cases", -1)) != len(dataset.data):
        raise ValueError("pseudo cache positive-case count mismatch")
    if marker.get("public_read") is not False or marker.get("hidden_read") is not False:
        raise ValueError("pseudo cache provenance mismatch")
    return marker


def populate(dataset, workers: int, minimum_free_gib: float) -> tuple[int, int]:
    loader = DataLoader(
        CacheOnlyView(dataset),
        batch_size=None,
        shuffle=False,
        num_workers=workers,
        prefetch_factor=1 if workers else None,
        persistent_workers=bool(workers),
    )
    started = time.time()
    cached = 0
    created = 0
    for index, result in enumerate(loader, 1):
        if result["error"]:
            raise RuntimeError(
                f"pseudo cache failure for {result['name']}: {result['error']}"
            )
        cached += int(bool(result["cached"]))
        created += int(not bool(result["cached"]))
        if index == 1 or index % 25 == 0 or index == len(dataset):
            free_gib = shutil.disk_usage(dataset.cache_dir).free / GIB
            if free_gib < minimum_free_gib:
                raise RuntimeError(
                    f"pseudo cache free space below reserve: {free_gib:.1f} GiB"
                )
            rate = index / max(time.time() - started, 1.0)
            eta_hours = (len(dataset) - index) / max(rate, 1e-9) / 3600.0
            print(
                f"[pseudo-cache] {index}/{len(dataset)} created={created} "
                f"reused={cached} rate={rate:.3f} cases/s eta={eta_hours:.2f}h "
                f"free={free_gib:.1f}GiB",
                flush=True,
            )
    return created, cached


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--minimum-free-gib", type=float, default=100.0)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 0:
        raise ValueError("workers must be non-negative")
    if not math.isfinite(args.minimum_free_gib) or args.minimum_free_gib < 1:
        raise ValueError("minimum-free-gib must be finite and positive")

    cache_dir = args.cache_dir.resolve()
    base_marker = cache_dir / "PREPROCESSING_COMPLETE.json"
    if not base_marker.is_file():
        raise ValueError("base preprocessing completion marker is missing")
    base = json.loads(base_marker.read_text(encoding="utf-8"))
    if base.get("spacing") != [1.0, 1.0, 3.0] or base.get("roi") != [96, 96, 64]:
        raise ValueError("base preprocessing geometry mismatch")
    contract = json.loads(args.dataset_contract.read_text(encoding="utf-8"))
    expected = validate_contract(contract, args.dataset_contract, args.manifest)
    dataset = pseudo_dataset(cache_dir, args.manifest)
    if len(dataset.data) != expected:
        raise ValueError(
            f"pseudo cache dataset count mismatch: {len(dataset.data)} != {expected}"
        )

    if args.verify_only:
        marker = verify_marker(
            args.output, dataset, args.manifest, args.dataset_contract, base_marker
        )
        print(json.dumps({"status": "PASS", "marker": marker}, indent=2))
        return

    abdomen_dir = cache_dir / "abdomen"
    temporary_dir = Path(tempfile.gettempdir()).resolve()
    temporary_dir.mkdir(parents=True, exist_ok=True)
    if temporary_dir.stat().st_dev != abdomen_dir.stat().st_dev:
        raise RuntimeError(
            "MONAI cache temporary directory is not on the cache filesystem; "
            "cross-filesystem shutil.move is not atomic"
        )
    sizes = [path.stat().st_size for path in abdomen_dir.glob("*.pt") if path.stat().st_size > 0]
    if not sizes:
        raise ValueError("base abdomen cache is empty")
    mean_bytes = sum(sizes) / len(sizes)
    free_before = shutil.disk_usage(abdomen_dir).free
    projected_bytes = int(mean_bytes * expected * 1.25)
    reserve_bytes = int(args.minimum_free_gib * GIB)
    if free_before < projected_bytes + reserve_bytes:
        raise RuntimeError(
            "insufficient pseudo-cache capacity: "
            f"free={free_before/GIB:.1f}GiB projected={projected_bytes/GIB:.1f}GiB "
            f"reserve={args.minimum_free_gib:.1f}GiB"
        )

    created, reused = populate(dataset, args.workers, args.minimum_free_gib)
    inventory = cache_inventory(dataset)
    payload = {
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "COMPLETE",
        "cache_dir": str(cache_dir),
        "temporary_dir": str(temporary_dir),
        "cache_device": int(abdomen_dir.stat().st_dev),
        "temporary_device": int(temporary_dir.stat().st_dev),
        "atomic_publication": "same_filesystem_temporary_then_rename",
        "positive_cases": expected,
        "created_entries": created,
        "reused_entries": reused,
        "workers": args.workers,
        "minimum_free_gib": args.minimum_free_gib,
        "free_before_bytes": free_before,
        "free_after_bytes": shutil.disk_usage(abdomen_dir).free,
        "projected_bytes_with_25pct_margin": projected_bytes,
        "base_abdomen_mean_cache_bytes": mean_bytes,
        "cache_inventory": inventory,
        "pseudo_manifest": asset(args.manifest),
        "dataset_contract": asset(args.dataset_contract),
        "base_cache_marker": asset(base_marker),
        "spacing": [1.0, 1.0, 3.0],
        "roi": [96, 96, 64],
        "public_read": False,
        "hidden_read": False,
        "submission_created": False,
        "uploaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    verify_marker(args.output, dataset, args.manifest, args.dataset_contract, base_marker)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
