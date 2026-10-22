from pathlib import Path

import numpy as np
import pytest
import torch

from analyze_component_classifier_cv import load_component_embeddings, model_grid
from extract_glance_component_embeddings import (
    centered_crop,
    normalize_checkpoint_state,
)


def test_centered_crop_preserves_center_and_pads_edges():
    source = np.arange(5 * 6 * 7, dtype=np.float32).reshape(5, 6, 7)
    middle = centered_crop(source, (2, 3, 3), (3, 3, 3))
    assert middle.shape == (3, 3, 3)
    assert middle[1, 1, 1] == source[2, 3, 3]

    edge = centered_crop(source, (0, 0, 0), (4, 4, 4))
    assert edge.shape == (4, 4, 4)
    assert edge[2, 2, 2] == source[0, 0, 0]
    assert np.count_nonzero(edge[:2]) == 0


def test_checkpoint_prefix_normalization_is_glance_only():
    state = {
        "module.glance_model.backbone.conv1.weight": torch.ones(1),
        "module.segmentation_model.weight": torch.zeros(1),
    }
    normalized = normalize_checkpoint_state(state)
    assert list(normalized) == ["backbone.conv1.weight"]


def test_embedding_archive_requires_exact_component_order(tmp_path: Path):
    path = tmp_path / "features.npz"
    np.savez_compressed(
        path,
        cases=np.asarray(["case_a", "case_b"]),
        component_ids=np.asarray([2, 7]),
        embeddings=np.asarray([[1, 2], [3, 4]], dtype=np.float16),
        logits=np.asarray([[0.1, 0.4], [0.5, -0.2]], dtype=np.float32),
    )
    rows = [
        {"case": "case_a", "component_id": 2},
        {"case": "case_b", "component_id": 7},
    ]
    features, metadata = load_component_embeddings(path, rows)
    assert features.shape == (2, 3)
    np.testing.assert_allclose(features[:, -1], [0.3, -0.7], atol=1e-6)
    assert metadata["embedding_dimensions"] == 2

    with pytest.raises(ValueError, match="identity mismatch"):
        load_component_embeddings(path, list(reversed(rows)))


def test_deep_grid_uses_only_regularized_linear_models():
    names = [name for name, _model in model_grid(7, deep_features=True)]
    assert len(names) == 4
    assert all(name.startswith("deep_logistic") for name in names)

