"""Small inference-only fallback for :mod:`connected_components_3d`.

The required challenge image already contains SciPy but not ``cc3d``.  The
deployment recipe only needs binary 3-D connected-component labels, so using
``scipy.ndimage.label`` avoids adding a compiled wheel to the image.  Keep the
surface deliberately narrow and fail closed if training/analysis code asks for
an unsupported cc3d feature.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
from scipy import ndimage


def connected_components(
    data: np.ndarray,
    *,
    connectivity: int = 26,
    return_N: bool = False,
    **kwargs,
) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
    """Label a binary 3-D mask with cc3d-compatible connectivity semantics."""

    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise TypeError(f"unsupported cc3d fallback arguments: {unsupported}")
    array = np.asarray(data)
    if array.ndim != 3:
        raise ValueError("cc3d deployment fallback only supports 3-D arrays")
    if array.dtype != np.bool_ and not np.isin(np.unique(array), (0, 1)).all():
        raise ValueError("cc3d deployment fallback only supports binary arrays")
    if connectivity == 6:
        structure = ndimage.generate_binary_structure(3, 1)
    elif connectivity == 18:
        structure = ndimage.generate_binary_structure(3, 2)
    elif connectivity == 26:
        structure = ndimage.generate_binary_structure(3, 3)
    else:
        raise ValueError("connectivity must be one of 6, 18, or 26")
    labels, count = ndimage.label(array.astype(bool, copy=False), structure=structure)
    labels = labels.astype(np.uint32, copy=False)
    return (labels, int(count)) if return_N else labels


def statistics(*args, **kwargs):
    raise NotImplementedError(
        "cc3d.statistics is intentionally unavailable in the inference-only fallback"
    )
