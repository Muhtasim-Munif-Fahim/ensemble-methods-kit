"""Tests for balanced_sample_weights utility."""
from __future__ import annotations

import numpy as np
import pytest

from ensemble_methods_kit import balanced_sample_weights


def test_balanced_weights_sum_to_equal_class_totals() -> None:
    y = np.array([0, 0, 0, 1, 1])
    weights = balanced_sample_weights(y)
    assert weights.shape == (5,)
    assert abs(weights[y == 0].sum() - weights[y == 1].sum()) < 1e-6


def test_balanced_weights_inverse_frequency() -> None:
    y = np.array([0, 0, 0, 1])
    weights = balanced_sample_weights(y)
    assert weights[3] == pytest.approx(weights[0] * 3)


def test_balanced_weights_multiclass() -> None:
    y = np.array([0, 1, 2, 0, 1, 2, 0])
    weights = balanced_sample_weights(y)
    assert weights.shape == (7,)
    w0 = weights[y == 0]
    w1 = weights[y == 1]
    w2 = weights[y == 2]
    assert abs(w0.sum() - w1.sum()) < 1e-6
    assert abs(w1.sum() - w2.sum()) < 1e-6


def test_balanced_weights_single_class_raises() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        balanced_sample_weights(np.array([0, 0, 0]))


def test_balanced_weights_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        balanced_sample_weights(np.array([]))


def test_balanced_weights_reproducible() -> None:
    y = np.array([0, 1, 0, 1, 0, 1, 2])
    a = balanced_sample_weights(y)
    b = balanced_sample_weights(y)
    np.testing.assert_array_equal(a, b)


def test_balanced_weights_mean_proportional_to_inverse_frequency() -> None:
    y = np.array([0, 0, 0, 0, 1, 1, 2])
    weights = balanced_sample_weights(y)
    expected = np.array([7 / 12, 7 / 12, 7 / 12, 7 / 12, 7 / 6, 7 / 6, 7 / 3])
    np.testing.assert_allclose(weights, expected, atol=1e-6)
