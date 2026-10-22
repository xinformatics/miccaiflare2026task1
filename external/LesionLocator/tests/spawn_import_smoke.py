"""Regression smoke for the OpenMP configuration used by preprocessing workers."""

from __future__ import annotations

import multiprocessing
import os


def child(queue):
    import numpy  # noqa: F401
    import scipy  # noqa: F401
    import torch  # noqa: F401

    queue.put(os.environ.get("MKL_THREADING_LAYER"))


if __name__ == "__main__":
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=child, args=(queue,))
    process.start()
    process.join(30)
    if process.exitcode != 0:
        raise SystemExit(f"spawned import worker failed with exit code {process.exitcode}")
    if queue.get(timeout=5) != "GNU":
        raise SystemExit("spawned worker did not inherit MKL_THREADING_LAYER=GNU")
    print("spawned NumPy/SciPy/PyTorch import passed with GNU OpenMP")
