#!/usr/bin/env python3
"""Sequential, hash-bound inference for the FLARE25 partial-box fine-tune."""

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
from nnunet.flare25_partialbox_finetune import (  # noqa: E402
    nnUNetTrainerFLARE25PartialBoxFineTune,
)


TRAINER_NAME = "nnUNetTrainerFLARE25PartialBoxFineTune"
safe.TRAINER_CLASSES[TRAINER_NAME] = nnUNetTrainerFLARE25PartialBoxFineTune


def terminal_argument(argv: list[str]) -> Path:
    try:
        index = argv.index("--terminal")
        value = argv[index + 1]
    except (ValueError, IndexError) as error:
        raise ValueError("FLARE25 fine-tune inference requires --terminal") from error
    return Path(value)


def rewrite_terminal(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETE" or payload.get("trainer") != TRAINER_NAME:
        raise ValueError("FLARE25 fine-tune inference terminal trainer mismatch")
    payload["model"] = "official FLARE25 PlainConvUNet partial-box fine-tune"
    payload["architecture_parameters"] = 31_195_594
    payload["partial_label_training_policy"] = (
        "ignore_outside_union_of_16mm_component_boxes"
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

