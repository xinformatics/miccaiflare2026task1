from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from nnunet.build_dataset504_flare25_multiteacher import select_gf_components


def test_keeps_cross_teacher_component_and_not_unmatched_small_component():
    gf = np.zeros((20, 20, 20), dtype=np.uint8)
    gf[1:5, 1:5, 1:5] = 1
    gf[10:12, 10:12, 10:12] = 1
    flare = np.zeros_like(gf)
    flare[2:5, 2:5, 2:5] = 1
    selected, rows = select_gf_components(gf, flare, 1.0)
    assert selected.sum() == 64
    assert [row["reason"] for row in rows] == ["cross_teacher_consensus", "unknown"]


def test_large_component_recall_rescue_is_not_erased_by_confirmer():
    gf = np.zeros((40, 40, 40), dtype=np.uint8)
    gf[1:33, 1:33, 1:33] = 1
    selected, rows = select_gf_components(gf, np.zeros_like(gf), 1.0)
    assert selected.sum() == 32**3
    assert rows[0]["reason"] == "large_component_recall_rescue"


def test_shape_and_voxel_volume_contracts_are_enforced():
    gf = np.zeros((4, 4, 4), dtype=np.uint8)
    try:
        select_gf_components(gf, np.zeros((3, 4, 4), dtype=np.uint8), 1.0)
    except ValueError as error:
        assert "aligned" in str(error)
    else:
        raise AssertionError("shape mismatch was accepted")
    try:
        select_gf_components(gf, gf, 0.0)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("nonpositive voxel volume was accepted")
