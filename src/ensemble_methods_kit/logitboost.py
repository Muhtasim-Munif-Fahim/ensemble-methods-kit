"""Additive logistic boosting: LogitBoost and Gentle AdaBoost.

Friedman, Hastie & Tibshirani (2000), *Additive logistic regression: a
statistical view of boosting*, explain AdaBoost as stagewise fitting of an
additive model ``F(x)`` under exponential loss. They also introduce two
variants that fit weighted least-squares regression trees, not classifiers:

* **LogitBoost** (their Algorithm 3 and the J-class Algorithm 6) takes
  Newton steps on the multinomial log-likelihood. For each class ``j``, with
  ``p_j = softmax(F)_j``, it fits a regression tree to the working response
  ``z_j = (y_j - p_j) / (p_j (1 - p_j))`` with weights ``w_j = p_j (1 - p_j)``.
  The class contributions are then centred,
  ``f_j <- (J - 1)/J (f_j - mean_k f_k)``, so they sum to zero. As the paper
  recommends, ``z`` is clipped to ``[-z_max, z_max]`` and weights are floored
  for numerical stability.
* **Gentle AdaBoost** (Algorithm 4) fits ``y in {-1, +1}`` by weighted least
  squares with ``w = exp(-y F)``. It adds the bounded tree output (shrunk by
  ``learning_rate``) to ``F``. The steps are smaller and more stable than
  Real or Discrete AdaBoost. Multiclass problems use a one-vs-rest
  reduction. Probabilities come from ``p = 1 / (1 + exp(-2 F))``, the
  population minimiser of exponential loss, normalised across classes.

Both models grow their own small weighted regression trees, because the
kit's CART tree has no sample weights. Split search uses prefix sums of
``w`` and ``w z``.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

__all__ = ["WeightedRegressionTree", "LogitBoostClassifier", "GentleBoostClassifier"]


class WeightedRegressionTree:
    """Depth-limited weighted least-squares regression tree.

    Each split maximises the weighted SSE reduction
    ``S_L^2 / W_L + S_R^2 / W_R - S^2 / W``, where ``S = sum w z`` and
    ``W = sum w``. Leaves predict the weighted mean ``S / W``.
    """

    def __init__(self, max_depth: int = 1, min_samples_leaf: int = 1) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1")
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self._nodes: list = []

    def fit(self, X, z, sample_weight=None) -> "WeightedRegressionTree":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        z = np.asarray(z, dtype=np.float64).ravel()
        w = np.ones_like(z) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64).ravel()
        if X.shape[0] != z.shape[0] or w.shape != z.shape:
            raise ValueError("X, z and sample_weight must have matching lengths")
        if np.any(w < 0) or not np.any(w > 0):
            raise ValueError("sample_weight must be non-negative with a positive sum")
        self.n_features_in_ = X.shape[1]
        self._nodes = []
        self._grow(X, z, w, np.arange(X.shape[0]), 0)
        return self

    def _leaf(self, z, w, idx) -> int:
        W = w[idx].sum()
        value = float(np.dot(w[idx], z[idx]) / W) if W > 0 else 0.0
        self._nodes.append(("leaf", value))
        return len(self._nodes) - 1

    def _best_split(self, X, z, w, idx):
        best = (0.0, None, None)
        m = idx.size
        k = self.min_samples_leaf
        if m < 2 * k:
            return best
        wz_all = w[idx] * z[idx]
        S, W = wz_all.sum(), w[idx].sum()
        base = S * S / W if W > 0 else 0.0
        for j in range(X.shape[1]):
            order = np.argsort(X[idx, j], kind="mergesort")
            xs = X[idx[order], j]
            cw = np.cumsum(w[idx[order]])[:-1]
            cs = np.cumsum(wz_all[order])[:-1]
            wl, wr = cw, W - cw
            valid = (xs[1:] > xs[:-1]) & (wl > 1e-300) & (wr > 1e-300)
            pos = np.arange(1, m)
            valid &= (pos >= k) & (m - pos >= k)
            if not valid.any():
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                gain = cs ** 2 / wl + (S - cs) ** 2 / wr - base
            gain = np.where(valid, gain, -np.inf)
            i = int(np.argmax(gain))
            if gain[i] > best[0] + 1e-12:
                best = (float(gain[i]), j, 0.5 * (xs[i] + xs[i + 1]))
        return best

    def _grow(self, X, z, w, idx, depth) -> int:
        if depth >= self.max_depth:
            return self._leaf(z, w, idx)
        gain, j, thr = self._best_split(X, z, w, idx)
        if j is None:
            return self._leaf(z, w, idx)
        node = len(self._nodes)
        self._nodes.append(None)
        mask = X[idx, j] <= thr
        left = self._grow(X, z, w, idx[mask], depth + 1)
        right = self._grow(X, z, w, idx[~mask], depth + 1)
        self._nodes[node] = ("split", j, thr, left, right)
        return node

    def predict(self, X) -> np.ndarray:
        if not self._nodes:
            raise RuntimeError("tree is not fitted")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        out = np.empty(X.shape[0])
        stack = [(0, np.arange(X.shape[0]))]
        while stack:
            n, rows = stack.pop()
            node = self._nodes[n]
            if node[0] == "leaf":
                out[rows] = node[1]
                continue
            _, j, thr, left, right = node
            mask = X[rows, j] <= thr
            stack.append((left, rows[mask]))
            stack.append((right, rows[~mask]))
        return out


def _check_common(n_estimators, learning_rate, max_depth, min_samples_leaf):
    if n_estimators < 1:
        raise ValueError("n_estimators must be at least 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")
    if min_samples_leaf < 1:
        raise ValueError("min_samples_leaf must be at least 1")


def _prepare_xy(X, y):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y).ravel()
    if X.shape[0] != y.shape[0] or X.shape[0] == 0:
        raise ValueError("X and y must be non-empty with matching lengths")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must be finite")
    classes, y_idx = np.unique(y, return_inverse=True)
    if classes.size < 2:
        raise ValueError("need at least two classes")
    return X, classes, y_idx


def _softmax(F):
    F = F - F.max(axis=1, keepdims=True)
    e = np.exp(F)
    return e / e.sum(axis=1, keepdims=True)


class LogitBoostClassifier:
    """LogitBoost: Newton boosting of the multinomial log-likelihood.

    Parameters
    ----------
    n_estimators :
        Number of boosting rounds. Each round fits one tree per class.
    learning_rate :
        Shrinkage applied to every round's contribution.
    max_depth :
        Depth of the weighted regression trees (``1`` gives stumps).
    min_samples_leaf :
        Minimum number of samples in a leaf.
    z_max :
        Clip on the working response (Friedman et al. suggest 2 to 4).

    After fitting, ``train_loss_`` holds the mean training log-loss before
    the first round and after each round.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.5,
        max_depth: int = 1,
        min_samples_leaf: int = 1,
        z_max: float = 4.0,
    ) -> None:
        _check_common(n_estimators, learning_rate, max_depth, min_samples_leaf)
        if z_max <= 0:
            raise ValueError("z_max must be positive")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.z_max = z_max
        self.estimators_: List[List[WeightedRegressionTree]] = []
        self.classes_: Optional[np.ndarray] = None
        self.train_loss_: List[float] = []

    def fit(self, X, y) -> "LogitBoostClassifier":
        X, self.classes_, y_idx = _prepare_xy(X, y)
        n, J = X.shape[0], self.classes_.size
        Y = np.eye(J)[y_idx]
        F = np.zeros((n, J))
        self.estimators_ = []
        self.train_loss_ = [float(np.log(J))]
        for _ in range(self.n_estimators):
            P = _softmax(F)
            round_trees, f = [], np.zeros((n, J))
            for j in range(J):
                pj = P[:, j]
                w = np.maximum(pj * (1.0 - pj), 1e-10)
                zj = np.clip((Y[:, j] - pj) / w, -self.z_max, self.z_max)
                tree = WeightedRegressionTree(self.max_depth, self.min_samples_leaf).fit(X, zj, w)
                f[:, j] = tree.predict(X)
                round_trees.append(tree)
            f = (J - 1.0) / J * (f - f.mean(axis=1, keepdims=True))
            F += self.learning_rate * f
            self.estimators_.append(round_trees)
            P = _softmax(F)
            self.train_loss_.append(float(-np.mean(np.log(np.maximum(P[np.arange(n), y_idx], 1e-300)))))
        return self

    def _check_fitted(self, X):
        if not self.estimators_:
            raise RuntimeError("model is not fitted")
        X = np.asarray(X, dtype=np.float64)
        return X.reshape(-1, 1) if X.ndim == 1 else X

    def staged_decision_function(self, X):
        """Yield the additive scores ``F(x)`` after each boosting round."""
        X = self._check_fitted(X)
        J = self.classes_.size
        F = np.zeros((X.shape[0], J))
        for trees in self.estimators_:
            f = np.column_stack([t.predict(X) for t in trees])
            F = F + self.learning_rate * (J - 1.0) / J * (f - f.mean(axis=1, keepdims=True))
            yield F

    def decision_function(self, X) -> np.ndarray:
        F = None
        for F in self.staged_decision_function(X):
            pass
        return F

    def predict_proba(self, X) -> np.ndarray:
        return _softmax(self.decision_function(X))

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def staged_predict(self, X):
        for F in self.staged_decision_function(X):
            yield self.classes_[np.argmax(F, axis=1)]


class GentleBoostClassifier:
    """Gentle AdaBoost with weighted least-squares regression trees.

    Binary problems fit one additive model. Multiclass problems fit one
    one-vs-rest model per class and predict the class with the largest
    ``F_k(x)``. ``predict_proba`` normalises ``1 / (1 + exp(-2 F_k))``.
    ``train_loss_`` tracks the mean exponential loss ``exp(-y F)``,
    averaged over the per-class models.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 1.0,
        max_depth: int = 1,
        min_samples_leaf: int = 1,
    ) -> None:
        _check_common(n_estimators, learning_rate, max_depth, min_samples_leaf)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.estimators_: List[List[WeightedRegressionTree]] = []
        self.classes_: Optional[np.ndarray] = None
        self.train_loss_: List[float] = []

    def fit(self, X, y) -> "GentleBoostClassifier":
        X, self.classes_, y_idx = _prepare_xy(X, y)
        J = self.classes_.size
        targets = [1] if J == 2 else list(range(J))
        Ys = [np.where(y_idx == k, 1.0, -1.0) for k in targets]
        Fs = [np.zeros(X.shape[0]) for _ in targets]
        self.estimators_ = []
        self.train_loss_ = [1.0]
        for _ in range(self.n_estimators):
            round_trees = []
            for yk, Fk in zip(Ys, Fs):
                w = np.exp(-yk * Fk)
                w /= w.sum()
                tree = WeightedRegressionTree(self.max_depth, self.min_samples_leaf).fit(X, yk, w)
                Fk += self.learning_rate * tree.predict(X)
                round_trees.append(tree)
            self.estimators_.append(round_trees)
            self.train_loss_.append(float(np.mean([np.mean(np.exp(-yk * Fk)) for yk, Fk in zip(Ys, Fs)])))
        return self

    def staged_decision_function(self, X):
        """Yield ``F`` after each round: shape ``(n,)`` if binary, else ``(n, J)``."""
        if not self.estimators_:
            raise RuntimeError("model is not fitted")
        X = np.asarray(X, dtype=np.float64)
        X = X.reshape(-1, 1) if X.ndim == 1 else X
        F = np.zeros((X.shape[0], len(self.estimators_[0])))
        for trees in self.estimators_:
            F = F + self.learning_rate * np.column_stack([t.predict(X) for t in trees])
            yield F[:, 0] if F.shape[1] == 1 else F

    def decision_function(self, X) -> np.ndarray:
        F = None
        for F in self.staged_decision_function(X):
            pass
        return F

    def predict_proba(self, X) -> np.ndarray:
        F = self.decision_function(X)
        if F.ndim == 1:
            p1 = 1.0 / (1.0 + np.exp(-2.0 * F))
            return np.column_stack([1.0 - p1, p1])
        P = 1.0 / (1.0 + np.exp(-2.0 * F))
        return P / P.sum(axis=1, keepdims=True)

    def _label(self, F):
        if F.ndim == 1:
            return self.classes_[(F > 0).astype(int)]
        return self.classes_[np.argmax(F, axis=1)]

    def predict(self, X) -> np.ndarray:
        return self._label(self.decision_function(X))

    def staged_predict(self, X):
        for F in self.staged_decision_function(X):
            yield self._label(F)
