#!/usr/bin/env python3
"""Populate GF-Screen's deterministic MONAI cache before DDP training.

Every input item is visited exactly once, so no two workers can write or
repair the same cache key.  Existing valid entries are reused.  A completion
marker is written only after every expected cache file exists.
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

from torch.utils.data import DataLoader, Dataset

from utils.data_utils import get_abdomen_ds, get_chest_ds, get_nolesion_ds


class CacheOnlyView(Dataset):
    def __init__(self, persistent_dataset):
        self.dataset = persistent_dataset

    def __len__(self):
        return len(self.dataset.data)

    def __getitem__(self, index):
        item = self.dataset.data[index]
        key = self.dataset.hash_func(item).decode("utf-8") + self.dataset.transform_hash
        cache_file = self.dataset.cache_dir / f"{key}.pt"
        # Files are written through a same-filesystem temporary directory and
        # atomically renamed.  On resume, avoid loading terabytes of completed
        # tensors merely to prove they exist again.
        if cache_file.is_file() and cache_file.stat().st_size > 0:
            return {"index": index, "name": item.get("name", key), "error": "", "cached": True}
        try:
            # _cachecheck applies only the deterministic prefix and atomically
            # writes it.  Do not return the volume through multiprocessing IPC.
            self.dataset._cachecheck(item)
            return {"index": index, "name": item.get("name", key), "error": "", "cached": False}
        except Exception as exc:
            return {
                "index": index,
                "name": item.get("name", key),
                "image": item.get("image", ""),
                "label": item.get("label", ""),
                "error": f"{type(exc).__name__}: {exc}",
                "cached": False,
            }


def unique_dataset(dataset):
    seen = set()
    unique = []
    for item in dataset.data:
        key = dataset.hash_func(item).decode("utf-8") + dataset.transform_hash
        if key not in seen:
            seen.add(key)
            unique.append(item)
    dataset.data = unique
    return dataset


def populate(name, dataset, workers, completed, total, started, failures):
    dataset = unique_dataset(dataset)
    loader = DataLoader(
        CacheOnlyView(dataset),
        batch_size=None,
        shuffle=False,
        num_workers=workers,
        prefetch_factor=1 if workers else None,
        persistent_workers=bool(workers),
    )
    for local_index, result in enumerate(loader, 1):
        done = completed + local_index
        if result["error"]:
            failure = {
                "group": name,
                "name": result["name"],
                "image": result["image"],
                "label": result["label"],
                "error": result["error"],
            }
            failures.append(failure)
            print(f"[FAILED] {json.dumps(failure, sort_keys=True)}", flush=True)
        if local_index == 1 or local_index % 25 == 0 or local_index == len(dataset):
            elapsed = max(time.time() - started, 1.0)
            rate = done / elapsed
            eta_hours = (total - done) / max(rate, 1e-9) / 3600.0
            free = shutil.disk_usage(dataset.cache_dir).free / (1024 ** 4)
            print(
                f"[{name}] {local_index}/{len(dataset)} overall={done}/{total} "
                f"rate={rate:.3f} cases/s eta={eta_hours:.2f}h free={free:.2f}TiB",
                flush=True,
            )
    return completed + len(dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--abdomen-workers", type=int)
    parser.add_argument("--chest-workers", type=int)
    parser.add_argument("--healthy-workers", type=int)
    parser.add_argument("--abdomen-json", default="jsons/local/abdomen.json")
    parser.add_argument("--chest-json", default="jsons/local/chest.json")
    parser.add_argument("--healthy-json", default="jsons/local/healthy.json")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    for region in ("abdomen", "chest", "healthy"):
        (cache_dir / region).mkdir(exist_ok=True)

    # Dataset construction consults this marker and deliberately omits prior
    # failures.  Preserve those names when refreshing the marker; otherwise a
    # fast resume scan would forget the omitted cases and reintroduce their
    # absent/corrupt cache keys into the following training process.
    marker_final = cache_dir / "PREPROCESSING_COMPLETE.json"
    prior_failures = []
    prior_failed_names = set()
    if marker_final.is_file():
        try:
            prior_marker = json.loads(marker_final.read_text())
            prior_failures = list(prior_marker.get("failures", []))
            prior_failed_names = set(prior_marker.get("failed_names", []))
        except (OSError, ValueError, TypeError):
            prior_failures = []
            prior_failed_names = set()

    transform_args = SimpleNamespace(
        data_dir=".",
        abdomen_json=args.abdomen_json,
        chest_json=args.chest_json,
        healthy_json=args.healthy_json,
        healthy_repeat=1,
        use_persistent_dataset=True,
        cache_dir=str(cache_dir),
        space_x=1.0,
        space_y=1.0,
        space_z=3.0,
        b_min=0.0,
        b_max=1.0,
        roi_x=96,
        roi_y=96,
        roi_z=64,
        sw_batch_size=2,
        # The crop is the first random transform and is not persisted, but the
        # current data pipeline still needs its constructor arguments while
        # composing the deterministic prefix.
        cancer_positive_weight=1.0,
        cancer_negative_weight=0.0,
        component_uniform_crop=False,
        component_crop_points_per_lesion=64,
        component_crop_size_power=0.0,
        component_background_points=0,
        component_crop_cache_dir="",
        component_legacy_cache_dir="",
        RandFlipd_prob=0.2,
        RandRotate90d_prob=0.2,
        RandShiftIntensityd_prob=0.1,
    )

    abdomen_train, abdomen_val = get_abdomen_ds(transform_args)
    chest_train, chest_val = get_chest_ds(transform_args)
    healthy_train, healthy_val = get_nolesion_ds(transform_args)
    groups = [
        ("abdomen-train", abdomen_train),
        ("abdomen-val", abdomen_val),
        ("chest-train", chest_train),
        ("chest-val", chest_val),
        ("healthy-train", healthy_train),
        ("healthy-val", healthy_val),
    ]
    groups = [(name, unique_dataset(dataset)) for name, dataset in groups]
    total = sum(len(dataset) for _, dataset in groups)
    print(f"Offline preprocessing {total} unique split entries into {cache_dir}", flush=True)

    started = time.time()
    completed = 0
    failures = []
    for name, dataset in groups:
        region = name.split("-", 1)[0]
        workers = getattr(args, f"{region}_workers")
        if workers is None:
            workers = args.workers
        print(f"[{name}] using {workers} worker(s)", flush=True)
        completed = populate(name, dataset, workers, completed, total, started, failures)

    counts = {
        region: len(list((cache_dir / region).glob("*.pt")))
        for region in ("abdomen", "chest", "healthy")
    }
    current_failed_names = {failure["name"] for failure in failures}
    combined_failed_names = sorted(prior_failed_names | current_failed_names)
    combined_failures = {
        failure.get("name"): failure
        for failure in prior_failures + failures
        if isinstance(failure, dict) and failure.get("name")
    }
    marker = {
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "expected_entries": total,
        "cache_file_counts": counts,
        "spacing": [1.0, 1.0, 3.0],
        "roi": [96, 96, 64],
        "workers": {
            "default": args.workers,
            "abdomen": args.abdomen_workers if args.abdomen_workers is not None else args.workers,
            "chest": args.chest_workers if args.chest_workers is not None else args.workers,
            "healthy": args.healthy_workers if args.healthy_workers is not None else args.workers,
        },
        "failed_count": len(combined_failed_names),
        "failed_names": combined_failed_names,
        "failures": [combined_failures.get(name, {"name": name, "error": "preserved prior failure"})
                     for name in combined_failed_names],
    }
    marker_tmp = cache_dir / ".PREPROCESSING_COMPLETE.json.tmp"
    marker_tmp.write_text(json.dumps(marker, indent=2) + "\n")
    os.replace(marker_tmp, marker_final)
    print(f"COMPLETE marker={marker_final} counts={counts}", flush=True)


if __name__ == "__main__":
    main()
