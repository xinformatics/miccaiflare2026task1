import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from select_dsc_first_nonregression import evaluate


def run(candidate_dsc_gain: float, candidate_nsd_gain: float):
    anchor = np.tile(np.asarray([[0.5, 0.5]]), (50, 1))
    candidate = anchor + np.asarray([candidate_dsc_gain, candidate_nsd_gain])
    return evaluate(
        anchor,
        candidate,
        bootstrap=1000,
        seed=7,
        min_dsc_gain=0.003,
        min_dsc_win_rate=0.52,
        min_nsd_gain=0.0,
        min_nsd_uncertainty_bound=-0.005,
        max_catastrophic_dsc_rate=0.10,
        catastrophic_dsc_delta=-0.20,
    )


def test_promotes_dsc_gain_when_nsd_is_maintained():
    result = run(0.01, 0.0)
    assert result["decision"] == "PROMOTE"
    assert all(result["checks"].values())


def test_rejects_nsd_regression_even_with_dsc_gain():
    result = run(0.02, -0.01)
    assert result["decision"] == "KEEP_ANCHOR"
    assert not result["checks"]["nsd_mean_nonregression"]


def test_rejects_nsd_only_improvement_without_dsc_gain():
    result = run(0.0, 0.02)
    assert result["decision"] == "KEEP_ANCHOR"
    assert not result["checks"]["dsc_practical_gain"]
    assert not result["checks"]["dsc_ci_positive"]
