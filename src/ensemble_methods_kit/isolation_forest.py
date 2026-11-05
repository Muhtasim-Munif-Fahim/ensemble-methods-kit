"""Isolation Forest for unsupervised anomaly detection (Liu, Ting & Zhou, 2008).

An isolation tree recursively partitions a small subsample with random axis
splits (random feature, uniform threshold between the node's min and max)
until every point is isolated or a height limit ``ceil(log2(psi))`` is hit.
Anomalies are few and different, so they are isolated near the root: their
expected path length ``E[h(x)]`` over the ensemble is short.

The anomaly score normalises path length by ``c(psi)``, the average path
length of an unsuccessful BST search over ``psi`` points:

    s(x) = 2 ** (-E[h(x)] / c(psi)),   c(n) = 2 H(n - 1) - 2 (n - 1) / n,

so ``s -> 1`` for clear anomalies, ``s ~ 0.5`` or below for normal points.
Leaves that stop early at the height limit add ``c(leaf_size)`` to the path.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

__all__ = ["IsolationForest", "average_path_length"]

_EULER_GAMMA = 0.5772156649015329


def average_path_length(n: Union[int, np.ndarray]) -> np.ndarray:
    """``c(n)``: expected path length of an unsuccessful BST search over ``n`` points."""
    n = np.asarray(n, dtype=np.float64)
    out = np.zeros_like(n)
    out[n == 2] = 1.0
    big = n > 2
    m = n[big]
    out[big] = 2.0 * (np.log(m - 1.0) + _EULER_GAMMA) - 2.0 * (m - 1.0) / m
    return out


class _IsolationTree:
    """Array-backed isolation tree."""

    def __init__(self, X: np.ndarray, height_limit: int, rng: np.random.Generator) -> None:
        self.feature: list[int] = []
        self.threshold: list[float] = []
        self.left: list[int] = []
        self.right: list[int] = []
        self.size: list[int] = []
        self._build(X, 0, height_limit, rng)
        self.feature_ = np.asarray(self.feature)
        self.threshold_ = np.asarray(self.threshold)
        self.left_ = np.asarray(self.left)
        self.right_ = np.asarray(self.right)
        self.leaf_correction_ = average_path_length(np.asarray(self.size))

    def _new_node(self, size: int) -> int:
        self.feature.append(-1)
        self.threshold.append(0.0)
        self.left.append(-1)
        self.right.append(-1)
        self.size.append(size)
        return len(self.size) - 1

    def _build(self, X: np.ndarray, depth: int, limit: int, rng: np.random.Generator) -> int:
        node = self._new_node(X.shape[0])
        if depth >= limit or X.shape[0] <= 1:
            return node
        lo, hi = X.min(axis=0), X.max(axis=0)
        candidates = np.flatnonzero(hi > lo)
        if candidates.size == 0:
            return node
        f = int(rng.choice(candidates))
        t = float(rng.uniform(lo[f], hi[f]))
        mask = X[:, f] < t
        self.feature[node] = f
        self.threshold[node] = t
        self.left[node] = self._build(X[mask], depth + 1, limit, rng)
        self.right[node] = self._build(X[~mask], depth + 1, limit, rng)
        return node

    def path_length(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        node = np.zeros(n, dtype=np.int64)
        depth = np.zeros(n, dtype=np.float64)
        active = self.feature_[node] >= 0
        while active.any():
            idx = np.flatnonzero(active)
            cur = node[idx]
            go_left = X[idx, self.feature_[cur]] < self.threshold_[cur]
            node[idx] = np.where(go_left, self.left_[cur], self.right_[cur])
            depth[idx] += 1.0
            active = self.feature_[node] >= 0
        return depth + self.leaf_correction_[node]


class IsolationForest:
    """Isolation Forest anomaly detector.

    Parameters
    ----------
    n_estimators :
        Number of isolation trees.
    max_samples :
        Subsample size ``psi`` per tree (int) or fraction of ``n`` (float in
        ``(0, 1]``). The default 256 follows the original paper.
    contamination :
        Expected outlier fraction used to set ``threshold_`` for
        :meth:`predict`; ``None`` uses the paper's fixed cut ``s > 0.5``.
    random_state :
        Seed for subsampling and random splits.

    Attributes
    ----------
    estimators_ :
        The fitted isolation trees.
    max_samples_ :
        Effective subsample size.
    threshold_ :
        Score above which :meth:`predict` flags an anomaly.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: Union[int, float] = 256,
        contamination: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        if isinstance(max_samples, float) and not 0.0 < max_samples <= 1.0:
            raise ValueError("float max_samples must be in (0, 1]")
        if isinstance(max_samples, int) and max_samples < 2:
            raise ValueError("int max_samples must be >= 2")
        if contamination is not None and not 0.0 < contamination < 0.5:
            raise ValueError("contamination must be in (0, 0.5)")
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, X, y=None) -> "IsolationForest":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("X must be 2D with at least 2 rows")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values")
        n = X.shape[0]
        if isinstance(self.max_samples, float):
            psi = max(2, int(round(self.max_samples * n)))
        else:
            psi = int(self.max_samples)
        psi = min(psi, n)
        rng = np.random.default_rng(self.random_state)
        limit = int(np.ceil(np.log2(psi)))
        self.estimators_ = [
            _IsolationTree(X[rng.choice(n, size=psi, replace=False)], limit, rng)
            for _ in range(self.n_estimators)
        ]
        self.max_samples_ = psi
        self.n_features_in_ = X.shape[1]
        self._c = float(average_path_length(psi))
        if self.contamination is None:
            self.threshold_ = 0.5
        else:
            self.threshold_ = float(np.quantile(self.score_samples(X), 1.0 - self.contamination))
        return self

    def _check(self, X) -> np.ndarray:
        if not hasattr(self, "estimators_"):
            raise RuntimeError("IsolationForest is not fitted")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"X must have shape (n, {self.n_features_in_})")
        return X

    def path_lengths(self, X) -> np.ndarray:
        """Mean path length ``E[h(x)]`` across trees."""
        X = self._check(X)
        return np.mean([t.path_length(X) for t in self.estimators_], axis=0)

    def score_samples(self, X) -> np.ndarray:
        """Anomaly score ``s(x)`` in ``(0, 1]``; higher means more anomalous."""
        return np.power(2.0, -self.path_lengths(X) / self._c)

    def decision_function(self, X) -> np.ndarray:
        """``score - threshold_``: positive for predicted anomalies."""
        return self.score_samples(X) - self.threshold_

    def predict(self, X) -> np.ndarray:
        """Return ``1`` for anomalies and ``0`` for inliers."""
        return (self.decision_function(X) > 0).astype(int)

    def fit_predict(self, X, y=None) -> np.ndarray:
        return self.fit(X).predict(X)
