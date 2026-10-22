from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import screen_component_hysteresis_consensus as implementation


def config(**overrides):
    values = dict(
        name="test",
        max_mm3=2000,
        min_seed_fraction=0.10,
        min_seed_mm3=1,
        min_persistence_fraction=0.10,
        min_persistence_mm3=1,
        min_union_prompt_fraction=0,
        min_union_prompt_mm3=0,
        min_each_prompt_mm3=1,
        min_consensus_votes=0,
        grow_mm=float("inf"),
    )
    values.update(overrides)
    return implementation.Config(**values)


def test_anchor_is_immutable_and_supported_disconnected_component_is_added():
    shape = (12, 12, 12)
    base = np.zeros(shape, bool)
    base[1:3, 1:3, 1:3] = True
    candidate = np.zeros(shape, bool)
    candidate[1:3, 1:3, 1:3] = True  # Rejected because it touches the anchor.
    candidate[6:10, 6:10, 6:10] = True
    seed = np.zeros(shape, bool)
    seed[7:9, 7:9, 7:9] = True
    persistence = seed.copy()
    whole = seed.copy()
    tumor = seed.copy()
    features = implementation.component_features(
        base, candidate, seed, persistence, whole, tumor, (1, 1, 1)
    )
    prediction, chosen = implementation.materialize_prediction(
        base, seed, features, config(), (1, 1, 1)
    )
    assert len(chosen) == 1
    assert np.all(prediction[base])
    assert prediction[6:10, 6:10, 6:10].all()


def test_component_without_high_confidence_seed_is_rejected():
    shape = (10, 10, 10)
    empty = np.zeros(shape, bool)
    candidate = empty.copy()
    candidate[4:8, 4:8, 4:8] = True
    support = candidate.copy()
    features = implementation.component_features(
        empty, candidate, empty, support, support, support, (1, 1, 1)
    )
    prediction, chosen = implementation.materialize_prediction(
        empty, empty, features, config(), (1, 1, 1)
    )
    assert len(chosen) == 0
    assert not prediction.any()


def test_finite_hysteresis_growth_trims_low_confidence_boundary():
    shape = (15, 15, 15)
    empty = np.zeros(shape, bool)
    candidate = empty.copy()
    candidate[2:13, 6:9, 6:9] = True
    seed = empty.copy()
    seed[7:8, 7:8, 7:8] = True
    support = candidate.copy()
    features = implementation.component_features(
        empty, candidate, seed, candidate, support, support, (1, 1, 1)
    )
    prediction, chosen = implementation.materialize_prediction(
        empty,
        seed,
        features,
        config(min_seed_fraction=0, grow_mm=2),
        (1, 1, 1),
    )
    assert len(chosen) == 1
    assert prediction[7, 7, 7]
    assert not prediction[2, 7, 7]
    assert prediction.sum() < candidate.sum()
