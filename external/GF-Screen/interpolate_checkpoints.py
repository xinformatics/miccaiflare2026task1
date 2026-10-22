#!/usr/bin/env python3
"""Create data-free WiSE-FT style interpolations of compatible checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.25])
    parser.add_argument("--scopes", choices=("all", "focus"), nargs="+", default=["all", "focus"])
    parser.add_argument(
        "--allow-extrapolation",
        action="store_true",
        help="allow coefficients outside [0, 1] for a guarded endpoint probe",
    )
    args = parser.parse_args()

    anchor_checkpoint = torch.load(args.anchor, map_location="cpu", weights_only=False)
    adapted_checkpoint = torch.load(args.adapted, map_location="cpu", weights_only=False)
    anchor = anchor_checkpoint["state_dict"]
    adapted = adapted_checkpoint["state_dict"]
    if anchor.keys() != adapted.keys():
        missing = sorted(anchor.keys() - adapted.keys())
        extra = sorted(adapted.keys() - anchor.keys())
        raise ValueError(f"state dictionaries differ: missing={missing[:5]} extra={extra[:5]}")
    for key in anchor:
        if anchor[key].shape != adapted[key].shape:
            raise ValueError(f"shape mismatch for {key}: {anchor[key].shape} vs {adapted[key].shape}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for scope in args.scopes:
        for alpha in args.alphas:
            if not args.allow_extrapolation and not 0.0 <= alpha <= 1.0:
                raise ValueError(
                    f"alpha must be in [0, 1], got {alpha}; "
                    "pass --allow-extrapolation for an intentional endpoint probe"
                )
            state = {}
            blended = 0
            for key, anchor_value in anchor.items():
                in_scope = scope == "all" or key.startswith("segmentation_model.")
                adapted_value = adapted[key]
                if in_scope and (anchor_value.is_floating_point() or anchor_value.is_complex()):
                    state[key] = torch.lerp(anchor_value, adapted_value, alpha)
                    blended += 1
                elif in_scope and alpha >= 1.0:
                    state[key] = adapted_value.clone()
                else:
                    state[key] = anchor_value.clone()
            alpha_tag = f"{alpha:.3f}".replace(".", "p")
            output = args.output_dir / f"wise_{scope}_alpha{alpha_tag}.pt"
            torch.save({
                "state_dict": state,
                "epoch": anchor_checkpoint.get("epoch", 0),
                "best_acc": anchor_checkpoint.get("best_acc", 0.0),
                "interpolation": {
                    "anchor": str(args.anchor),
                    "adapted": str(args.adapted),
                    "alpha": alpha,
                    "scope": scope,
                    "blended_tensors": blended,
                },
            }, output)
            print(f"{output}: blended {blended} tensors", flush=True)


if __name__ == "__main__":
    main()
