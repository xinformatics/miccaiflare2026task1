import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from screen_boundary_postprocess import componentwise_transform, physical_ball  # noqa: E402


def test_physical_ball_respects_anisotropic_spacing():
    ball = physical_ball((0.5, 0.5, 3.0), 1.0)
    assert ball.shape == (5, 5, 3)
    assert ball[:, :, 1].sum() > 1
    assert not ball[:, :, 0].any()
    assert not ball[:, :, 2].any()


def test_fill_holes_does_not_join_components_or_add_distant_foreground():
    mask = np.zeros((12, 12, 12), dtype=bool)
    mask[1:6, 1:6, 1:6] = True
    mask[2:5, 2:5, 2:5] = False
    mask[9:11, 9:11, 9:11] = True
    result = componentwise_transform(mask, (1, 1, 1), "fill_holes_3d")
    assert result[3, 3, 3]
    assert not result[7, 7, 7]
    assert result[9:11, 9:11, 9:11].all()


def test_identity_is_exact():
    rng = np.random.default_rng(7)
    mask = rng.random((9, 8, 7)) > 0.8
    assert np.array_equal(componentwise_transform(mask, (1, 1, 1), "identity"), mask)
