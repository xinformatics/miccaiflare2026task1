#!/usr/bin/env python3
"""Best-effort Weights & Biases logging that can never stop training."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any


def _warn(action: str, error: Exception) -> None:
    print(
        f"WARNING: W&B {action} failed; continuing training without W&B: "
        f"{type(error).__name__}: {error}",
        file=sys.stderr,
        flush=True,
    )


def safe_wandb_init(**kwargs: Any) -> Any | None:
    """Initialize W&B, returning ``None`` when tracking is unavailable."""

    try:
        import wandb

        return wandb.init(**kwargs)
    except Exception as error:  # W&B/network state must not control training.
        _warn("initialization", error)
        return None


def stable_wandb_run_id(logdir: str | Path, name: str | None) -> str:
    """Return one deterministic W&B ID for all resumes of a run directory."""

    identity = f"{Path(logdir).resolve()}::{name or ''}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def safe_wandb_define_metrics(
    run: Any | None,
    definitions: list[tuple[str, dict[str, Any]]],
) -> Any | None:
    """Define metrics once and disable tracking if the client fails."""

    if run is None:
        return None
    try:
        for name, kwargs in definitions:
            run.define_metric(name, **kwargs)
        return run
    except Exception as error:
        _warn("metric definition", error)
        return None


def safe_wandb_log(
    run: Any | None,
    record: dict[str, Any],
    step: int,
    **kwargs: Any,
) -> Any | None:
    """Log once and disable the run after its first client/network failure."""

    if run is None:
        return None
    try:
        run.log(record, step=step, **kwargs)
        return run
    except Exception as error:  # Preserve checkpoints and optimizer progress.
        _warn("logging", error)
        return None


def safe_wandb_finish(run: Any | None) -> None:
    """Finish a W&B run without changing the process exit status."""

    if run is None:
        return
    try:
        run.finish()
    except Exception as error:
        _warn("finish", error)
