#!/usr/bin/env python3
"""Archive immutable epoch checkpoints from already-running nnU-Net jobs."""

from __future__ import annotations

import argparse
import gc
import logging
import os
from pathlib import Path
import shutil
import time

import torch


LOGGER = logging.getLogger("checkpoint_archiver")


def parse_run(value: str) -> tuple[str, Path]:
    try:
        label, path = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("run must be LABEL=CHECKPOINT_DIRECTORY") from error
    directory = Path(path).resolve()
    if not label or not directory.is_dir():
        raise argparse.ArgumentTypeError(f"invalid run: {value}")
    return label, directory


def wait_until_stable(path: Path, interval: float = 2.0) -> os.stat_result:
    previous = path.stat()
    while True:
        time.sleep(interval)
        current = path.stat()
        if (
            current.st_mtime_ns == previous.st_mtime_ns
            and current.st_size == previous.st_size
            and current.st_size > 0
        ):
            return current
        previous = current


def completed_epoch(path: Path) -> int:
    while True:
        try:
            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
            epoch = int(checkpoint["current_epoch"]) - 1
            del checkpoint
            gc.collect()
            if epoch < 0:
                raise ValueError(f"invalid completed epoch {epoch} in {path}")
            return epoch
        except (EOFError, OSError, RuntimeError, KeyError) as error:
            LOGGER.warning("Checkpoint is not readable yet (%s); retrying", error)
            time.sleep(2)


def archive(label: str, directory: Path) -> int:
    latest = directory / "checkpoint_latest.pth"
    stable = wait_until_stable(latest)
    epoch = completed_epoch(latest)
    destination = directory / f"checkpoint_epoch_{epoch:04d}.pth"
    if destination.exists():
        if destination.stat().st_size != stable.st_size:
            raise RuntimeError(f"Existing archive has the wrong size: {destination}")
        LOGGER.info("%s epoch %d already archived", label, epoch)
        return stable.st_mtime_ns

    temporary = directory / f".{destination.name}.{os.getpid()}.tmp"
    try:
        shutil.copy2(latest, temporary)
        if temporary.stat().st_size != stable.st_size:
            raise RuntimeError(f"Incomplete checkpoint copy: {temporary}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    LOGGER.info(
        "%s archived epoch %d (%0.2f GiB) to %s",
        label,
        epoch,
        destination.stat().st_size / 2**30,
        destination,
    )
    return stable.st_mtime_ns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--pid-file", type=Path)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        filename=args.log_file,
    )
    if args.pid_file:
        args.pid_file.write_text(f"{os.getpid()}\n")

    seen: dict[str, int] = {}
    try:
        while True:
            for label, directory in args.run:
                latest = directory / "checkpoint_latest.pth"
                current = latest.stat().st_mtime_ns
                if seen.get(label) != current:
                    seen[label] = archive(label, directory)
            if args.once:
                return
            time.sleep(args.poll_seconds)
    finally:
        if args.pid_file:
            args.pid_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
