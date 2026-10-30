from __future__ import annotations

import numpy as np
import pytest

from proseometry.distribution import (
    featurization_is_safe_in_process,
    mauve_available,
    mauve_from_features,
)


def test_import_order_check() -> None:
    assert featurization_is_safe_in_process(["numpy", "proseometry"])
    assert featurization_is_safe_in_process(["torch", "faiss", "sklearn"])
    assert not featurization_is_safe_in_process(["faiss", "torch"])
    assert not featurization_is_safe_in_process(["sklearn"])
    assert not featurization_is_safe_in_process(["faiss", "sklearn", "numpy"])


@pytest.mark.skipif(not mauve_available(), reason="mauve-text not installed")
def test_isolated_clustering_separates_distributions() -> None:
    rng = np.random.default_rng(0)
    human = rng.normal(size=(200, 16))
    same = mauve_from_features(human, rng.normal(size=(200, 16)), seed=1, num_buckets=10)
    shifted = mauve_from_features(human, rng.normal(size=(200, 16)) + 4.0, seed=1, num_buckets=10)
    assert 0.0 <= shifted.mauve < 0.2 < 0.7 < same.mauve <= 1.0
    assert same.n_human == same.n_model == 200
    assert same.divergence_curve is not None and same.divergence_curve.shape[1] == 2
    again = mauve_from_features(human, rng.normal(size=(200, 16)), seed=1, num_buckets=10)
    assert isinstance(again.frontier_integral, float)


def test_feature_shape_validation() -> None:
    if not mauve_available():
        pytest.skip("mauve-text not installed")
    with pytest.raises(ValueError):
        mauve_from_features(np.zeros((5, 3)), np.zeros((5, 4)))
