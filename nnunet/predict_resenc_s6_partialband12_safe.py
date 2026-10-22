#!/usr/bin/env python3
"""Sequential, checkpoint-bound inference for the partial-band ResEnc S6."""

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
from nnunet.resenc_s6_partialband12 import (  # noqa: E402
    nnUNetTrainerResEncS6PanCancerPartialBand12,
)


TRAINER_NAME = "nnUNetTrainerResEncS6PanCancerPartialBand12"
safe.TRAINER_CLASSES[TRAINER_NAME] = nnUNetTrainerResEncS6PanCancerPartialBand12


def terminal_argument(argv: list[str]) -> Path:
    try:
        index = argv.index("--terminal")
        value = argv[index + 1]
    except (ValueError, IndexError) as error:
        raise ValueError("partial-band ResEnc inference requires --terminal") from error
    return Path(value)


def rewrite_terminal(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETE" or payload.get("trainer") != TRAINER_NAME:
        raise ValueError("partial-band ResEnc inference terminal trainer mismatch")
    payload["model"] = "nnU-Net v2 ResidualEncoderUNet S6 partial-band pan-cancer"
    payload["architecture_parameters"] = 107_467_050
    payload["partial_label_training_policy"] = "shape_linf_band_12mm"
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
