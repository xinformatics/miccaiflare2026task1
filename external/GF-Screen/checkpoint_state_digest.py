#!/usr/bin/env python3
"""Print a stable SHA-256 digest of checkpoint model tensors only."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


def state_dict_digest(path: Path) -> str:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict) or not state:
        raise ValueError(f"{path} does not contain a non-empty state_dict")

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"state_dict entry {name!r} is not a tensor")
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        # Flatten first so scalar integer buffers (for example
        # num_batches_tracked) can also be viewed as raw bytes.
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    print(state_dict_digest(args.checkpoint))


if __name__ == "__main__":
    main()
