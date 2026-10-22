from pathlib import Path
import sys

import torch
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from nnunet.flare25_multiteacher_distill import FrozenEncoderCosineScheduler
from nnunet.resenc_s6_positive_unlabeled import positive_unlabeled_sampling_probabilities


def test_differential_schedule_freezes_then_unfreezes_encoder():
    first = torch.nn.Parameter(torch.ones(1))
    second = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([
        {"params": [first], "lr": 0.0},
        {"params": [second], "lr": 1e-5},
    ])
    scheduler = FrozenEncoderCosineScheduler(optimizer)
    assert scheduler.rates(0) == (0.0, 1e-5)
    assert scheduler.rates(9) == (0.0, 1e-5)
    assert np.allclose(scheduler.rates(10), (1e-5, 3e-5), rtol=0, atol=1e-12)
    end = scheduler.rates(100)
    assert np.allclose(end, (1e-6, 3e-6), rtol=0, atol=1e-12)


def test_ten_percent_pseudo_exposure_is_exact():
    identifiers = ["Liver_001", "Liver_002", "Pseudo-AMOS_0001"]
    probabilities = positive_unlabeled_sampling_probabilities(
        identifiers, [100, 200, 300], pseudo_exposure=0.10
    )
    assert abs(probabilities[-1] - 0.10) < 1e-12
    assert abs(probabilities[:-1].sum() - 0.90) < 1e-12
