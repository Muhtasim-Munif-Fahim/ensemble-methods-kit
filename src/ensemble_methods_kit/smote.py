"""SMOTE oversampling and SMOTEBoost.

SMOTE (Chawla, Bowyer, Hall & Kegelmeyer, 2002) creates synthetic minority
samples by interpolating between a minority point and one of its ``k``
nearest minority-class neighbours:

    x_new = x_i + u * (x_nn - x_i),   u ~ Uniform(0, 1).

SMOTEBoost (Chawla, Lazarevic, Hall & Bowyer, 2003) injects a fresh batch of
SMOTE samples into every boosting round. The weak learner is trained on the
(weight-resampled) original data plus the synthetic points, while the
boosting error and the weight update are computed on the original training
set only, so synthetic points never accumulate weight. The original paper
uses AdaBoost.M2; this implementation uses multi-class SAMME (as
``RUSBoostClassifier`` does) so both imbalance boosters share the same
weighting scheme.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .utils import DecisionTree

__all__ = ["smote", "SMOTEBoostClassifier"]


def _validate_strategy(sampling_strategy) -> object:
    msg = 'sampling_strategy must be "auto" or a positive float'
    if isinstance(sampling_strategy, str):
        if sampling_strategy != "auto":
            raise ValueError(msg)
        return sampling_strategy
    if isinstance(sampling_strategy, bool) or not isinstance(sampling_strategy, (int, float)):
        raise ValueError(msg)
    if not np.isfinite(sampling_strategy) or float(sampling_strategy) <= 0.0:
        raise ValueError(msg)
    return float(sampling_strategy)


def _n_synthetic(y: np.ndarray, classes: np.ndarray, counts: np.ndarray, strategy) -> dict:
    """Number of synthetic samples to create for each class.

    ``"auto"`` raises every class to the majority count. A float ``r`` adds
    ``round(r * n_c)`` samples to every non-majority class ``c`` (``r=1.0`` is
    SMOTE's classic "N = 100%").
    """
    n_max = int(counts.max())
    out = {}
    for c, n_c in zip(classes, counts):
        if n_c == n_max:
            continue
        if strategy == "auto":
            out[c] = int(n_max - n_c)
        else:
            out[c] = int(round(float(strategy) * int(n_c)))
    return out


def _smote_class(X_c: np.ndarray, n_new: int, k: int, rng: np.random.Generator) -> np.ndarray:
    n_c = X_c.shape[0]
    if n_new <= 0:
        return np.empty((0, X_c.shape[1]))
    if n_c < 2:
        # No neighbour to interpolate with: replicate the single point.
        return np.repeat(X_c, n_new, axis=0)
    k_eff = min(k, n_c - 1)
    sq = np.sum(X_c * X_c, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * X_c @ X_c.T
    np.fill_diagonal(d2, np.inf)
    neighbours = np.argsort(d2, axis=1, kind="stable")[:, :k_eff]
    base = rng.integers(0, n_c, size=n_new)
    pick = neighbours[base, rng.integers(0, k_eff, size=n_new)]
    gap = rng.random((n_new, 1))
    return X_c[base] + gap * (X_c[pick] - X_c[base])


def smote(
    X,
    y,
    k_neighbors: int = 5,
    sampling_strategy="auto",
    random_state: Optional[int] = None,
    return_synthetic_only: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Oversample minority classes with SMOTE.

    Parameters
    ----------
    X, y :
        Training data and labels (binary or multi-class).
    k_neighbors :
        Number of same-class nearest neighbours used for interpolation
        (clipped to ``n_class - 1``).
    sampling_strategy :
        ``"auto"`` (balance every class to the majority size) or a positive
        float ``r`` adding ``round(r * n_c)`` synthetic samples per
        non-majority class.
    random_state :
        Seed.
    return_synthetic_only :
        If ``True`` return only the synthetic ``(X_syn, y_syn)``; otherwise the
        original data with the synthetic rows appended.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values")
    if isinstance(k_neighbors, bool) or not isinstance(k_neighbors, (int, np.integer)) or k_neighbors < 1:
        raise ValueError("k_neighbors must be a positive integer")
    strategy = _validate_strategy(sampling_strategy)
    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 2:
        raise ValueError("SMOTE needs at least two classes")
    rng = np.random.default_rng(random_state)
    plan = _n_synthetic(y, classes, counts, strategy)
    xs, ys = [], []
    for c in classes:
        n_new = plan.get(c, 0)
        if n_new <= 0:
            continue
        xs.append(_smote_class(X[y == c], n_new, int(k_neighbors), rng))
        ys.append(np.full(n_new, c, dtype=y.dtype))
    X_syn = np.vstack(xs) if xs else np.empty((0, X.shape[1]))
    y_syn = np.concatenate(ys) if ys else np.empty(0, dtype=y.dtype)
    if return_synthetic_only:
        return X_syn, y_syn
    return np.vstack([X, X_syn]), np.concatenate([y, y_syn])


class SMOTEBoostClassifier:
    """SMOTEBoost: SMOTE oversampling inside SAMME AdaBoost.

    Parameters
    ----------
    n_estimators :
        Maximum boosting rounds.
    learning_rate :
        Shrinkage on each weak learner's SAMME weight.
    max_depth :
        Depth of each weak tree (``1`` = stump).
    k_neighbors :
        SMOTE neighbourhood size.
    sampling_strategy :
        Passed to :func:`smote` each round (``"auto"`` or positive float).
    random_state :
        Seed for SMOTE and the weighted resampling.

    Attributes
    ----------
    estimators_, estimator_weights_, estimator_errors_, classes_, n_classes_.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        max_depth: Optional[int] = 1,
        k_neighbors: int = 5,
        sampling_strategy="auto",
        random_state: Optional[int] = None,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if isinstance(k_neighbors, bool) or not isinstance(k_neighbors, (int, np.integer)) or k_neighbors < 1:
            raise ValueError("k_neighbors must be a positive integer")
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = max_depth
        self.k_neighbors = int(k_neighbors)
        self.sampling_strategy = _validate_strategy(sampling_strategy)
        self.random_state = random_state
        self.estimators_: list = []
        self.estimator_weights_: list = []
        self.estimator_errors_: list = []
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: Optional[int] = None

    def fit(self, X, y) -> "SMOTEBoostClassifier":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        self.classes_ = np.unique(y)
        K = int(self.classes_.shape[0])
        if K < 2:
            raise ValueError("SMOTEBoost needs at least two classes")
        self.n_classes_ = K
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        w = np.full(n, 1.0 / n)
        self.estimators_, self.estimator_weights_, self.estimator_errors_ = [], [], []

        for _ in range(self.n_estimators):
            X_syn, y_syn = smote(
                X, y, k_neighbors=self.k_neighbors, sampling_strategy=self.sampling_strategy,
                random_state=int(rng.integers(0, 2**31 - 1)), return_synthetic_only=True,
            )
            boot = rng.choice(n, size=n, replace=True, p=w)
            X_train = np.vstack([X[boot], X_syn])
            y_train = np.concatenate([y[boot], y_syn])
            tree = DecisionTree(
                criterion="gini", max_depth=self.max_depth, min_samples_split=2,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            tree.fit(X_train, y_train)

            incorrect = tree.predict(X) != y
            err = float(np.dot(w, incorrect))
            if err >= 1.0 - 1.0 / K:
                if not self.estimators_:
                    # Keep one learner so the model is usable, with zero weight.
                    self.estimators_.append(tree)
                    self.estimator_weights_.append(0.0)
                    self.estimator_errors_.append(err)
                break
            clipped = min(max(err, 1e-16), 1.0 - 1e-16)
            alpha = self.learning_rate * (np.log((1.0 - clipped) / clipped) + np.log(K - 1.0))
            self.estimators_.append(tree)
            self.estimator_weights_.append(float(alpha))
            self.estimator_errors_.append(err)
            if err <= 0.0:
                break
            w = w * np.exp(alpha * incorrect.astype(float))
            w = np.clip(w, 1e-16, None)
            w /= w.sum()
        return self

    def decision_function(self, X) -> np.ndarray:
        if not self.estimators_:
            raise RuntimeError("SMOTEBoostClassifier must be fitted before predict")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        scores = np.zeros((X.shape[0], self.n_classes_))
        for tree, alpha in zip(self.estimators_, self.estimator_weights_):
            idx = np.searchsorted(self.classes_, tree.predict(X))
            scores[np.arange(X.shape[0]), idx] += alpha
        return scores

    def predict_proba(self, X) -> np.ndarray:
        scores = self.decision_function(X)
        total = float(np.sum(self.estimator_weights_))
        if total > 0.0:
            scores = scores / total
        scores = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]
