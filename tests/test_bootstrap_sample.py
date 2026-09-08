"""Tests for bootstrap_sample utility."""
from __future__ import annotations

import numpy as np
import pytest

from ensemble_methods_kit import bootstrap_sample


def _make_data(n: int = 10) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n, 2))
    y = np.arange(n)
    return X, y


def test_returns_correct_shape() -> None:
    X, y = _make_data(10)
    Xb, yb = bootstrap_sample(X, y, n_samples=5, random_state=0)
    assert Xb.shape == (5, 2)
    assert yb.shape == (5,)


def test_preserves_x_y_pairing() -> None:
    X, y = _make_data(10)
    Xb, yb = bootstrap_sample(X, y, n_samples=100, random_state=1)
    for i in range(len(yb)):
        assert yb[i] in y
        idx = np.where(y == yb[i])[0][0]
        np.testing.assert_allclose(Xb[i], X[idx])


def test_reproducible_with_seed() -> None:
    X, y = _make_data(20)
    a = bootstrap_sample(X, y, n_samples=15, random_state=42)
    b = bootstrap_sample(X, y, n_samples=15, random_state=42)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_without_replacement_uses_all_original_rows() -> None:
    X, y = _make_data(8)
    Xb, yb = bootstrap_sample(X, y, replace=False, random_state=0)
    assert len(yb) == 8
    assert sorted(yb.tolist()) == sorted(y.tolist())


def test_mismatched_dimensions_raise() -> None:
    X = np.zeros((5, 2))
    y = np.zeros(4)
    with pytest.raises(ValueError, match="same number of rows"):
        bootstrap_sample(X, y)


def test_empty_dataset_raises() -> None:
    X = np.zeros((0, 2))
    y = np.array([])
    with pytest.raises(ValueError, match="empty"):
        bootstrap_sample(X, y)


def test_invalid_n_samples() -> None:
    X, y = _make_data(10)
    with pytest.raises(ValueError, match="n_samples"):
        bootstrap_sample(X, y, n_samples=0)
    with pytest.raises(ValueError, match="n_samples"):
        bootstrap_sample(X, y, n_samples=1.5)


def test_no_replacement_exceeding_size_raises() -> None:
    X, y = _make_data(5)
    with pytest.raises(ValueError, match="replace=False"):
        bootstrap_sample(X, y, n_samples=10, replace=False)


def test_with_replacement_can_exceed_size() -> None:
    X, y = _make_data(5)
    Xb, yb = bootstrap_sample(X, y, n_samples=20, replace=True, random_state=0)
    assert Xb.shape[0] == 20


def test_default_n_samples_equals_input_size() -> None:
    X, y = _make_data(12)
    Xb, yb = bootstrap_sample(X, y, random_state=0)
    assert Xb.shape[0] == 12
