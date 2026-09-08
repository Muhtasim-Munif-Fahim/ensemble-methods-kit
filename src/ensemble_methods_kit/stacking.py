"""Stacking and blending meta-ensembles plus a logistic-regression oracle."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .utils import clone_estimator

__all__ = ["LogisticRegression", "StackingClassifier", "BlendingClassifier"]


def _softmax_matrix(scores: np.ndarray) -> np.ndarray:
    scores = scores - scores.max(axis=1, keepdims=True)
    e = np.exp(scores)
    return e / e.sum(axis=1, keepdims=True)


def _kfold_indices(n: int, k: int, rng: np.random.Generator) -> List[Tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    splits = []
    for i in range(k):
        va = folds[i]
        tr = np.concatenate([f for j, f in enumerate(folds) if j != i])
        splits.append((tr, va))
    return splits


class LogisticRegression:
    """Multinomial logistic regression trained by full-batch gradient descent.

    Used as the default meta-learner for stacking and blending.  Features are
    standardised internally so the optimiser is well-conditioned.

    Parameters
    ----------
    lr :
        Learning rate.
    n_iter :
        Number of gradient-descent epochs.
    l2 :
        L2 regularisation strength.
    random_state :
        Seed (controls nothing stochastically here but kept for API symmetry).
    """

    def __init__(
        self,
        lr: float = 0.5,
        n_iter: int = 400,
        l2: float = 1.0,
        random_state: Optional[int] = None,
    ) -> None:
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2
        self.random_state = random_state
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X, y) -> "LogisticRegression":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        class_to_int = {c: i for i, c in enumerate(self.classes_)}
        y_int = np.array([class_to_int[v] for v in y], dtype=np.intp)
        n, d = X.shape
        K = self.classes_.shape[0]
        self._mu = X.mean(axis=0)
        self._sigma = X.std(axis=0) + 1e-8
        Xs = (X - self._mu) / self._sigma
        onehot = np.zeros((n, K))
        onehot[np.arange(n), y_int] = 1.0
        self.W_ = np.zeros((d, K))
        self.b_ = np.zeros(K)
        for _ in range(self.n_iter):
            p = _softmax_matrix(Xs @ self.W_ + self.b_)
            grad_W = (Xs.T @ (p - onehot)) / n + (self.l2 * self.W_) / n
            grad_b = (p - onehot).mean(axis=0)
            self.W_ -= self.lr * grad_W
            self.b_ -= self.lr * grad_b
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Xs = (X - self._mu) / self._sigma
        return _softmax_matrix(Xs @ self.W_ + self.b_)

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class StackingClassifier:
    """Stack ensemble that learns a meta-model over base predictions.

    Base learners are evaluated by cross-validation so the meta-features
    feeding the meta-learner are out-of-sample and unbiased.  The base
    learners are then re-fit on the full data for the final prediction.

    Parameters
    ----------
    estimators :
        List of ``(name, estimator)`` pairs.
    meta_estimator :
        Learner trained on the stacked probabilities.  Defaults to
        :class:`LogisticRegression`.
    cv :
        Number of cross-validation folds used to build meta-features.
    passthrough :
        If True, the original features are appended to the meta-features.
    n_jobs :
        Placeholder kept for sklearn-style API compatibility (serial only).
    random_state :
        Seed for the fold splits.
    """

    def __init__(
        self,
        estimators: Sequence[Tuple[str, object]],
        meta_estimator: Optional[object] = None,
        cv: int = 5,
        passthrough: bool = False,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.estimators = list(estimators)
        self.meta_estimator = meta_estimator
        self.cv = cv
        self.passthrough = passthrough
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.classes_: Optional[np.ndarray] = None
        self.meta_ = None
        self.base_full_: List = []

    def _default_meta(self):
        return LogisticRegression(lr=0.5, n_iter=400, l2=1.0, random_state=self.random_state)

    def fit(self, X, y) -> "StackingClassifier":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = self.classes_.shape[0]
        n_est = len(self.estimators)
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        meta = np.zeros((n, n_est * n_classes))
        for ei, (_, est) in enumerate(self.estimators):
            for tr, va in _kfold_indices(n, min(self.cv, n), rng):
                fold_est = clone_estimator(est)
                fold_est.fit(X[tr], y[tr])
                meta[va, ei * n_classes:(ei + 1) * n_classes] = fold_est.predict_proba(X[va])
        meta_X = meta
        if self.passthrough:
            meta_X = np.hstack([meta, X])
        meta_X_full = meta_X
        self.meta_ = self._default_meta() if self.meta_estimator is None else clone_estimator(self.meta_estimator)
        self.meta_.fit(meta_X_full, y)
        self.base_full_ = []
        for _, est in self.estimators:
            est_full = clone_estimator(est)
            est_full.fit(X, y)
            self.base_full_.append(est_full)
        return self

    def _meta_features(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        parts = [est.predict_proba(X) for est in self.base_full_]
        meta = np.hstack(parts)
        if self.passthrough:
            meta = np.hstack([meta, X])
        return meta

    def predict_proba(self, X) -> np.ndarray:
        return self.meta_.predict_proba(self._meta_features(X))

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.meta_.predict(self._meta_features(X))]


class BlendingClassifier:
    """Hold-out ensemble: meta-features come from a single validation split.

    Simpler and faster than stacking (no cross-validation) at the cost of
    slightly noisier meta-features.  Base learners are fit on the training
    partition; their validation predictions train the meta-learner.

    Parameters
    ----------
    estimators :
        List of ``(name, estimator)`` pairs.
    meta_estimator :
        Learner fit on the held-out predictions.  Defaults to
        :class:`LogisticRegression`.
    validation_fraction :
        Fraction of the training data reserved for building meta-features.
    passthrough :
        If True, append the raw features to the meta-features.
    random_state :
        Seed for the train/validation split.
    """

    def __init__(
        self,
        estimators: Sequence[Tuple[str, object]],
        meta_estimator: Optional[object] = None,
        validation_fraction: float = 0.3,
        passthrough: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        self.estimators = list(estimators)
        self.meta_estimator = meta_estimator
        self.validation_fraction = validation_fraction
        self.passthrough = passthrough
        self.random_state = random_state
        self.classes_: Optional[np.ndarray] = None
        self.meta_ = None
        self.base_full_: List = []

    def _default_meta(self):
        return LogisticRegression(lr=0.5, n_iter=400, l2=1.0, random_state=self.random_state)

    def fit(self, X, y) -> "BlendingClassifier":
        from .utils import train_test_split as _tts

        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        X_tr, X_va, y_tr, y_va = _tts(
            X, y, test_size=self.validation_fraction,
            random_state=self.random_state, stratify=y,
        )
        self.base_full_ = []
        meta_parts: List[np.ndarray] = []
        for _, est in self.estimators:
            est_full = clone_estimator(est)
            est_full.fit(X_tr, y_tr)
            meta_parts.append(est_full.predict_proba(X_va))
            self.base_full_.append(est_full)
        meta = np.hstack(meta_parts)
        if self.passthrough:
            meta = np.hstack([meta, X_va])
        self.meta_ = self._default_meta() if self.meta_estimator is None else clone_estimator(self.meta_estimator)
        self.meta_.fit(meta, y_va)
        # refit base learners on the full data for the final predictions
        self.base_full_ = []
        for _, est in self.estimators:
            est_full = clone_estimator(est)
            est_full.fit(X, y)
            self.base_full_.append(est_full)
        return self

    def _meta_features(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        parts = [est.predict_proba(X) for est in self.base_full_]
        meta = np.hstack(parts)
        if self.passthrough:
            meta = np.hstack([meta, X])
        return meta

    def predict_proba(self, X) -> np.ndarray:
        return self.meta_.predict_proba(self._meta_features(X))

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.meta_.predict(self._meta_features(X))]
