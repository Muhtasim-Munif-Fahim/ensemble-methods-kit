"""Breiman random-forest proximities and proximity-based outlier scores.

Two rows are *proximate* in a tree when they fall into the same leaf. The
forest proximity is the fraction of trees in which that happens::

    prox(i, j) = (1 / T) * sum_t 1{leaf_t(x_i) = leaf_t(x_j)}

It is a supervised similarity: rows are close when the trees, which were
grown to separate the labels, route them together. Breiman & Cutler use it
for clustering / MDS plots, prototypes and outlier detection. The outlier
measure of a training row is::

    raw_i = n / sum_{j in class(i), j != i} prox(i, j)^2

standardised within each class as ``(raw_i - median) / MAD`` (median absolute
deviation), so values well above ~10 flag rows that are far from the rest of
their class in the forest's eyes.

Works with any fitted ensemble whose ``estimators_`` are
:class:`~ensemble_methods_kit.utils.DecisionTree` instances (bagging, random
forests, extra trees, quantile forests, ...).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .utils import DecisionTree

__all__ = ["forest_apply", "proximity_matrix", "proximity_outlier_scores"]


def _trees(forest) -> list:
    trees = getattr(forest, "estimators_", None)
    if not trees:
        raise RuntimeError("forest must be fitted (non-empty estimators_)")
    for tree in trees:
        if not isinstance(tree, DecisionTree):
            raise TypeError("every member of estimators_ must be a DecisionTree")
    return list(trees)


def _as_2d(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("X must be 2-D")
    return X


def forest_apply(forest, X) -> np.ndarray:
    """Leaf index of every row in every tree, shape ``(n_samples, n_estimators)``."""
    X = _as_2d(X)
    return np.column_stack([tree.apply(X) for tree in _trees(forest)])


def proximity_matrix(forest, X, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """Forest proximity between rows of ``X`` and ``Y`` (default ``Y = X``).

    Returns an ``(n_X, n_Y)`` matrix with entries in ``[0, 1]``. For
    ``Y=None`` the matrix is symmetric with a unit diagonal.
    """
    leaves_x = forest_apply(forest, X)
    leaves_y = leaves_x if Y is None else forest_apply(forest, Y)
    if leaves_y.shape[1] != leaves_x.shape[1]:  # pragma: no cover - same forest
        raise ValueError("inconsistent number of trees")
    n_trees = leaves_x.shape[1]
    prox = np.zeros((leaves_x.shape[0], leaves_y.shape[0]), dtype=np.float64)
    for t in range(n_trees):
        prox += leaves_x[:, t][:, None] == leaves_y[:, t][None, :]
    return prox / n_trees


def proximity_outlier_scores(forest, X, y) -> np.ndarray:
    """Breiman's class-wise standardised outlier measure for labelled rows.

    Parameters
    ----------
    forest :
        A fitted tree ensemble (usually trained on ``X, y``).
    X, y :
        Labelled rows to score.

    Returns
    -------
    numpy.ndarray
        One score per row; larger means more outlying within its own class.
        Classes with a single member get ``0``.
    """
    X = _as_2d(X)
    y = np.asarray(y)
    if y.shape != (X.shape[0],):
        raise ValueError("y must be 1-D with one label per row of X")
    n = X.shape[0]
    prox = proximity_matrix(forest, X)
    np.fill_diagonal(prox, 0.0)
    scores = np.zeros(n, dtype=np.float64)
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        if idx.size < 2:
            continue
        sq = np.sum(prox[np.ix_(idx, idx)] ** 2, axis=1)
        raw = n / np.maximum(sq, 1e-12)
        med = np.median(raw)
        mad = np.median(np.abs(raw - med))
        if mad <= 0:
            mad = np.mean(np.abs(raw - med)) or 1.0
        scores[idx] = (raw - med) / mad
    return scores
