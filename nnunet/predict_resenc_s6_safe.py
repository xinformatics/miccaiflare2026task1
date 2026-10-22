#!/usr/bin/env python3
"""Use the proven sequential STU inference engine for corrected ResEnc S6.

This adapter changes only custom-trainer resolution and terminal metadata. The
underlying predictor retains checkpoint hashing, tensor-safe loading, in-
process preprocessing/export, compact uint16 probabilities, geometry checks,
dataset-role enforcement, and the absence of any upload capability.
"""

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
from nnunet.resenc_s6_partialbox import (  # noqa: E402
    nnUNetTrainerResEncS6PanCancerPartialBox,
)


TRAINER_NAME = "nnUNetTrainerResEncS6PanCancerPartialBox"
safe.TRAINER_CLASSES[TRAINER_NAME] = nnUNetTrainerResEncS6PanCancerPartialBox


def terminal_argument(argv: list[str]) -> Path:
    try:
        index = argv.index("--terminal")
        value = argv[index + 1]
    except (ValueError, IndexError) as error:
        raise ValueError("corrected ResEnc inference requires --terminal") from error
    return Path(value)


def rewrite_terminal(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETE" or payload.get("trainer") != TRAINER_NAME:
        raise ValueError("corrected ResEnc inference terminal trainer mismatch")
    payload["model"] = "nnU-Net v2 ResidualEncoderUNet S6 partial-box pan-cancer"
    payload["architecture_parameters"] = 107_467_050
    payload["partial_label_training_policy"] = (
        "ignore_outside_union_of_16mm_component_boxes"
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    terminal = terminal_argument(sys.argv[1:])
    # The shared predictor's fast reuse branch infers a trainer from a finite
    # STU-only model-directory name list. Force its ordinary checkpoint-bound
    # path instead. nnU-Net itself skips already complete outputs when
    # overwrite=False, after which the shared geometry/probability validator
    # still checks the exact shard.
    safe.reusable_compact_outputs = lambda cases, output: False
    safe.main()
    rewrite_terminal(terminal)


if __name__ == "__main__":
    main()

