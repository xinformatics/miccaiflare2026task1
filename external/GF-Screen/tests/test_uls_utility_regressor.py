import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


UTILITY = load("uls_utility", "analyze_uls_utility_regressor_cv.py")
FEATURE = load("uls_feature_for_utility_test", "analyze_uls_feature_classifier_cv.py")


def test_utility_target_matches_exact_single_addition_dice_change():
    candidate = FEATURE.Candidate(
        row={"case": "case", "addition_target_overlap": 15},
        addition_indices=np.arange(20),
    )
    stats = {
        "case": FEATURE.CaseStats(
            shape=(1, 1, 1),
            base_voxels=100,
            target_voxels=80,
            base_overlap=50,
            target_indices=np.empty(0, dtype=np.int64),
            reference=None,
        )
    }
    expected = 2 * 65 / 200 - 2 * 50 / 180
    assert np.isclose(UTILITY.utility_target(candidate, stats), expected)


def test_threshold_grid_is_nonnegative_sorted_and_has_zero():
    values = UTILITY.threshold_grid(np.asarray([-0.1, 0.01, 0.02, 0.5]))
    assert values == tuple(sorted(values))
    assert values[0] == 0.0
    assert all(value >= 0 for value in values)
