#!/usr/bin/env python3
"""Sequential inference adapter for the positive-unlabeled ResEnc S6."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PANCANCER_ROOT = PROJECT_ROOT / "external" / "STU-Net" / "pancancer"
for source_root in (PROJECT_ROOT, PANCANCER_ROOT):
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)

import predict_stunet_safe as safe  # noqa: E402
from nnunet.resenc_s6_positive_unlabeled import (  # noqa: E402
    nnUNetTrainerResEncS6PanCancerPositiveUnlabeled,
)


TRAINER_NAME = "nnUNetTrainerResEncS6PanCancerPositiveUnlabeled"
safe.TRAINER_CLASSES[TRAINER_NAME] = (
    nnUNetTrainerResEncS6PanCancerPositiveUnlabeled
)


def terminal_argument(argv: list[str]) -> Path:
    try:
        return Path(argv[argv.index("--terminal") + 1])
    except (ValueError, IndexError) as error:
        raise ValueError("positive ResEnc inference requires --terminal") from error


def rewrite_terminal(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETE" or payload.get("trainer") != TRAINER_NAME:
        raise ValueError("positive ResEnc inference terminal trainer mismatch")
    payload["model"] = (
        "nnU-Net v2 ResidualEncoderUNet S6 positive-unlabeled pan-cancer"
    )
    payload["architecture_parameters"] = 107_467_050
    payload["partial_label_training_policy"] = (
        "positive-only pseudo supervision; ignore outside component boxes"
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    terminal = terminal_argument(sys.argv[1:])
    safe.reusable_compact_outputs = lambda cases, output: False
    safe.main()
    rewrite_terminal(terminal)


if __name__ == "__main__":
    main()
