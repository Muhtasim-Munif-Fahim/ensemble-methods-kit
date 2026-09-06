"""Gradient boosting of regression / classification trees."""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

from .utils import DecisionTree

__all__ = ["GradientBoostingClassifier", "GradientBoostingRegressor"]


def _softmax(F: np.ndarray) -> np.ndarray:
    F = F - F.max(axis=1, keepdims=True)
    eF = np.exp(F)
    return eF / eF.sum(axis=1, keepdims=True)


class _BaseGradientBoosting:
    def __init__(self, n_estimators, learning_rate, max_depth,
                 min_samples_split, subsample, max_features, random_state):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.max_features = max_features
        self.random_state = random_state
        self.estimators_: List = []
        self._init = 0.0

    def _subsample_idx(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.subsample >= 1.0:
            return np.arange(n)
        n_sub = max(1, int(self.subsample * n))
        return rng.choice(n, n_sub, replace=False)


class GradientBoostingRegressor(_BaseGradientBoosting):
    """Gradient boosting regressor minimising squared error.

    Trees are fit to the per-sample residual ``y - F`` and added with the
    learning rate, producing the classic additive least-squares model.

    Parameters
    ----------
    n_estimators :
        Number of boosting stages (= number of trees).
    learning_rate :
        Shrinkage applied to each tree's contribution.
    max_depth :
        Maximum depth of each base regression tree.
    min_samples_split :
        Minimum samples required to split an internal node.
    subsample :
        Fraction of training rows sampled (without replacement) per stage.
        ``1.0`` uses the full dataset (deterministic boosting).
    max_features :
        Feature sub-sampling passed to each base tree.
    random_state :
        Seed for reproducible sub-sampling.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        subsample: float = 1.0,
        max_features: Optional[Union[int, str]] = None,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            subsample=subsample,
            max_features=max_features,
            random_state=random_state,
        )

    def fit(self, X, y) -> "GradientBoostingRegressor":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y, dtype=np.float64)
        n = X.shape[0]
        self._init = float(np.mean(y))
        F = np.full(n, self._init)
        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        for m in range(self.n_estimators):
            residual = y - F
            idx = self._subsample_idx(n, rng)
            tree = DecisionTree(
                criterion="variance",
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            tree.fit(X[idx], residual[idx])
            F += self.learning_rate * tree.predict(X)
            self.estimators_.append(tree)
        return self

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        F = np.full(X.shape[0], self._init)
        for tree in self.estimators_:
            F += self.learning_rate * tree.predict(X)
        return F


class GradientBoostingClassifier(_BaseGradientBoosting):
    """Gradient boosting classifier minimising multinomial deviance.

    Each boosting stage grows one regression tree per class whose target is the
    negative gradient of the (softmax) cross-entropy with respect to the
    current class scores.

    Parameters
    ----------
    n_estimators :
        Number of boosting stages.
    learning_rate :
        Shrinkage applied to each tree's contribution.
    max_depth :
        Maximum depth of each base tree.
    min_samples_split :
        Minimum samples required to split an internal node.
    subsample :
        Fraction of training rows sampled per stage.
    max_features :
        Feature sub-sampling passed to each base tree.
    random_state :
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        subsample: float = 1.0,
        max_features: Optional[Union[int, str]] = None,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            subsample=subsample,
            max_features=max_features,
            random_state=random_state,
        )

    def fit(self, X, y) -> "GradientBoostingClassifier":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        class_to_int = {c: i for i, c in enumerate(self.classes_)}
        y_int = np.array([class_to_int[v] for v in y], dtype=np.intp)
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        # initialise scores with the log-prior so softmax gives the class priors
        counts = np.bincount(y_int, minlength=self.n_classes_)
        probs = counts / n
        self._log_prior = np.log(probs + 1e-12)
        F = np.tile(self._log_prior, (n, 1))
        self.estimators_: List[List[DecisionTree]] = []
        for m in range(self.n_estimators):
            prob = _softmax(F)
            stage: List[DecisionTree] = []
            for k in range(self.n_classes_):
                residual = prob[:, k].copy()
                residual[y_int == k] -= 1.0  # p_k - 1{y=k} => negative gradient
                idx = self._subsample_idx(n, rng)
                tree = DecisionTree(
                    criterion="variance",
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split,
                    max_features=self.max_features,
                    random_state=int(rng.integers(0, 2**31 - 1)),
                )
                tree.fit(X[idx], -residual[idx])
                F[:, k] += self.learning_rate * tree.predict(X)
                stage.append(tree)
            self.estimators_.append(stage)
        return self

    def _decision(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n = X.shape[0]
        F = np.tile(self._log_prior, (n, 1))
        for stage in self.estimators_:
            for k, tree in enumerate(stage):
                F[:, k] += self.learning_rate * tree.predict(X)
        return F

    def predict_proba(self, X) -> np.ndarray:
        return _softmax(self._decision(X))

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]
