#!/usr/bin/env python3
"""Restore the exact epoch-105 Glance state into a Focus-adapted checkpoint."""

import argparse
from pathlib import Path

import torch


p = argparse.ArgumentParser()
p.add_argument("--base", type=Path, required=True)
p.add_argument("--adapted", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()

base = torch.load(a.base, map_location="cpu", weights_only=False)
adapted = torch.load(a.adapted, map_location="cpu", weights_only=False)
keys = [key for key in base["state_dict"] if key.startswith("glance_model.")]
for key in keys:
    adapted["state_dict"][key] = base["state_dict"][key].clone()
a.output.parent.mkdir(parents=True, exist_ok=True)
torch.save(adapted, a.output)
print(f"restored {len(keys)} Glance tensors -> {a.output}")
