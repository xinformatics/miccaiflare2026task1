#!/usr/bin/env python3
"""Measure a full-patch W48 forward/backward pass on one GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parent.parent
    plans = json.loads(
        (project / "nnUNet_preprocessed/Dataset501_PanCancer/nnUNetPlans.json").read_text()
    )
    config = plans["configurations"]["3d_fullres_WinnerW48"]
    architecture = config["architecture"]
    network = get_network_from_plans(
        architecture["network_class_name"],
        architecture["arch_kwargs"],
        architecture["_kw_requires_import"],
        input_channels=1,
        output_channels=2,
        allow_init=True,
        deep_supervision=True,
    ).cuda()
    optimizer = torch.optim.SGD(network.parameters(), lr=1e-3, momentum=0.99)
    patch = tuple(config["patch_size"])
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    data = torch.randn((args.batch, 1, *patch), device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        outputs = network(data)
        loss = sum(output.float().square().mean() for output in outputs)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    print(f"batch={args.batch} patch={patch} loss={loss.item():.6f}")
    print(f"allocated_gib={torch.cuda.max_memory_allocated() / 2**30:.3f}")
    print(f"reserved_gib={torch.cuda.max_memory_reserved() / 2**30:.3f}")


if __name__ == "__main__":
    main()
