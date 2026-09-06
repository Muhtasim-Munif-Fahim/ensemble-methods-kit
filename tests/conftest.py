"""Shared pytest fixtures generating deterministic synthetic datasets."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def binary_cls():
    """Two well-separated Gaussian blobs in 3D (120 per class)."""
    rng = np.random.default_rng(0)
    n = 120
    X = np.r_[rng.normal(0, 1, (n, 3)), rng.normal(4, 1, (n, 3))]
    y = np.r_[np.zeros(n, dtype=int), np.ones(n, dtype=int)]
    idx = rng.permutation(2 * n)
    return X[idx], y[idx]


@pytest.fixture
def multiclass_cls():
    """Three well-separated Gaussian blobs in 4D (100 per class)."""
    rng = np.random.default_rng(1)
    k, n, d = 3, 100, 4
    Xs = [rng.normal(i * 3, 1, (n, d)) for i in range(k)]
    X = np.vstack(Xs)
    y = np.concatenate([np.full(n, i, dtype=int) for i in range(k)])
    idx = rng.permutation(3 * n)
    return X[idx], y[idx]


@pytest.fixture
def regression():
    """Linear regression data with Gaussian noise (200 samples, 5 features)."""
    rng = np.random.default_rng(2)
    n, d = 200, 5
    X = rng.normal(0, 1, (n, d))
    w = rng.normal(0, 1, d)
    y = X @ w + 0.5 * rng.normal(0, 1, n)
    return X, y


@pytest.fixture
def split_cls(binary_cls):
    from ensemble_methods_kit import train_test_split

    X, y = binary_cls
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
