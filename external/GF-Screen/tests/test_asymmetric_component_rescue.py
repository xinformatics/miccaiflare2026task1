import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from asymmetric_component_rescue import apply_asymmetric_rescue  # noqa: E402


def test_primary_support_always_keeps_component():
    base = np.zeros((8, 8, 8), dtype=bool)
    base[1:3, 1:3, 1:3] = True
    primary = np.zeros_like(base)
    primary[1, 1, 1] = True
    secondary = np.zeros_like(base)
    result = apply_asymmetric_rescue(
        base, primary, secondary, (1.0, 1.0, 1.0), {"min_overlap_mm3": 30.0}
    )
    assert np.array_equal(result, base)


def test_secondary_threshold_rescues_only_supported_component():
    base = np.zeros((10, 10, 10), dtype=bool)
    base[1:4, 1:4, 1:4] = True
    base[6:9, 6:9, 6:9] = True
    primary = np.zeros_like(base)
    secondary = np.zeros_like(base)
    secondary[1:3, 1:3, 1:3] = True
    result = apply_asymmetric_rescue(
        base, primary, secondary, (2.0, 2.0, 2.0), {"min_overlap_mm3": 30.0}
    )
    assert result[1:4, 1:4, 1:4].all()
    assert not result[6:9, 6:9, 6:9].any()


def test_rescue_never_adds_secondary_only_foreground():
    base = np.zeros((8, 8, 8), dtype=bool)
    base[1:3, 1:3, 1:3] = True
    primary = np.zeros_like(base)
    secondary = np.ones_like(base)
    result = apply_asymmetric_rescue(base, primary, secondary, (1, 1, 1), {})
    assert np.array_equal(result, base)
