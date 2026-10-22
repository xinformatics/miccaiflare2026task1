#!/usr/bin/env python3
"""Summarize module-wise parameter movement between two GF-Screen checkpoints."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch


def load_state(path: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    metadata = {}
    for key in ("epoch", "best_acc"):
        if isinstance(checkpoint, dict) and key in checkpoint:
            value = checkpoint[key]
            if hasattr(value, "item"):
                value = value.item()
            metadata[key] = value
    return checkpoint.get("state_dict", checkpoint), metadata


def group_for(key: str) -> str:
    parts = key.split(".")
    if key.startswith("glance_model.backbone."):
        return "glance/" + parts[2]
    if key.startswith("glance_model.layer."):
        return "glance/head"
    if key.startswith("segmentation_model.swinViT."):
        return "focus/swinViT/" + parts[2]
    if key.startswith("segmentation_model."):
        return "focus/" + parts[1]
    return "other/" + parts[0]


def blank():
    return {
        "tensors": 0,
        "elements": 0,
        "changed_elements": 0,
        "reference_sq": 0.0,
        "candidate_sq": 0.0,
        "difference_sq": 0.0,
        "dot": 0.0,
    }


def update(stats, reference, candidate):
    stats["tensors"] += 1
    stats["elements"] += reference.numel()
    stats["changed_elements"] += int(torch.count_nonzero(reference != candidate))
    if reference.is_floating_point() or reference.is_complex():
        a = reference.detach().float().reshape(-1)
        b = candidate.detach().float().reshape(-1)
        d = b - a
        stats["reference_sq"] += float(torch.dot(a, a))
        stats["candidate_sq"] += float(torch.dot(b, b))
        stats["difference_sq"] += float(torch.dot(d, d))
        stats["dot"] += float(torch.dot(a, b))


def finish(stats):
    ref = math.sqrt(stats.pop("reference_sq"))
    cand = math.sqrt(stats.pop("candidate_sq"))
    diff = math.sqrt(stats.pop("difference_sq"))
    dot = stats.pop("dot")
    stats["l2_reference"] = ref
    stats["l2_candidate"] = cand
    stats["l2_difference"] = diff
    stats["relative_l2"] = diff / ref if ref else None
    stats["cosine_similarity"] = dot / (ref * cand) if ref and cand else None
    stats["changed_fraction"] = stats["changed_elements"] / stats["elements"]
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reference, reference_meta = load_state(args.reference)
    candidate, candidate_meta = load_state(args.candidate)
    if reference.keys() != candidate.keys():
        missing = sorted(set(reference) - set(candidate))
        extra = sorted(set(candidate) - set(reference))
        raise SystemExit(f"State dictionaries differ: missing={missing}, extra={extra}")

    groups = defaultdict(blank)
    for key, a in reference.items():
        b = candidate[key]
        if a.shape != b.shape:
            raise SystemExit(f"Shape mismatch for {key}: {a.shape} != {b.shape}")
        top = "glance" if key.startswith("glance_model.") else "focus"
        categories = ["all", top, group_for(key)]
        if key.endswith(("running_mean", "running_var", "num_batches_tracked")):
            categories += [top + "/normalization_buffers"]
            if key.endswith("num_batches_tracked"):
                categories += [top + "/normalization_counters"]
            else:
                categories += [top + "/normalization_running_stats"]
        else:
            categories += [top + "/parameters"]
        for category in categories:
            update(groups[category], a, b)

    result = {
        "reference": str(Path(args.reference).resolve()),
        "candidate": str(Path(args.candidate).resolve()),
        "reference_metadata": reference_meta,
        "candidate_metadata": candidate_meta,
        "groups": {name: finish(value) for name, value in sorted(groups.items())},
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
