"""Stable semantic proposal-source conditioning for candidate refiners."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from candidate_refiner_sampling import semantic_source_group


# Keep this order stable: it is serialized into every conditioned checkpoint.
DEFAULT_SOURCE_CONDITION_GROUPS = (
    "epoch105",
    "currentgf",
    "autopet_weak",
    "lndb_weak",
    "lung_cells",
)


def validate_source_condition_groups(groups: Sequence[str]) -> tuple[str, ...]:
    resolved = tuple(str(value).strip() for value in groups)
    if not resolved or any(not value for value in resolved):
        raise ValueError("source-conditioning groups must be nonempty names")
    if len(resolved) != len(set(resolved)):
        raise ValueError("source-conditioning groups must be unique")
    return resolved


def source_condition_index(source_name: str, groups: Sequence[str]) -> int:
    resolved = validate_source_condition_groups(groups)
    semantic = semantic_source_group(source_name)
    try:
        return resolved.index(semantic)
    except ValueError as exc:
        raise ValueError(
            f"proposal source {source_name!r} maps to unknown semantic group "
            f"{semantic!r}; expected one of {resolved}"
        ) from exc


def append_source_condition_channels(
    image: torch.Tensor,
    source_name: str,
    groups: Sequence[str],
) -> torch.Tensor:
    """Append spatially constant one-hot channels to one ``(C,D,H,W)`` patch."""

    if image.ndim != 4:
        raise ValueError("one candidate image must have shape (C,D,H,W)")
    resolved = validate_source_condition_groups(groups)
    index = source_condition_index(source_name, resolved)
    condition = image.new_zeros((len(resolved), *image.shape[1:]))
    condition[index].fill_(1)
    return torch.cat((image, condition), dim=0)


def append_source_condition_batch(
    images: torch.Tensor,
    source_names: Sequence[str],
    groups: Sequence[str],
) -> torch.Tensor:
    """Append one-hot source volumes to a ``(B,C,D,H,W)`` image batch."""

    if images.ndim != 5:
        raise ValueError("candidate image batch must have shape (B,C,D,H,W)")
    if len(source_names) != images.shape[0]:
        raise ValueError("source-name batch does not align with images")
    resolved = validate_source_condition_groups(groups)
    condition = images.new_zeros(
        (images.shape[0], len(resolved), *images.shape[2:])
    )
    for batch_index, source_name in enumerate(source_names):
        condition[batch_index, source_condition_index(source_name, resolved)].fill_(1)
    return torch.cat((images, condition), dim=1)
