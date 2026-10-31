"""DART: Dropouts meet Multiple Additive Regression Trees.

Rashmi & Gilad-Bachrach (AISTATS 2015). Ordinary gradient boosting suffers
from *over-specialisation*: the first trees explain most of the signal and
later trees only patch a handful of residuals. DART applies dropout to the
ensemble. At every boosting stage:

1. With probability ``skip_drop`` no tree is dropped (a normal boosting
   step). Otherwise every existing stage is dropped independently with
   probability ``rate_drop``. With ``one_drop=True`` at least one stage is
   dropped whenever dropout is applied.
2. The new tree is fit to the negative gradient of the loss evaluated at
   the ensemble *without* the dropped stages.
3. Normalisation ("tree" mode, as in XGBoost/LightGBM): with ``k`` dropped
   stages and learning rate ``eta`` the new stage gets weight
   ``eta / (k + eta)`` and every dropped stage is rescaled by
   ``k / (k + eta)``. When nothing is dropped the new stage simply gets
   weight ``eta``, so ``rate_drop=0`` reduces exactly to gradient boosting.

Per-stage weights are stored in ``tree_weights_``.
"""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

from .gradient_boosting import _softmax
from .utils import DecisionTree

__all__ = ["DARTClassifier", "DARTRegressor"]


class _BaseDART:
    def __init__(self, n_estimators, learning_rate, max_depth, min_samples_split,
                 subsample, max_features, rate_drop, skip_drop, one_drop, random_state):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.max_features = max_features
        self.rate_drop = rate_drop
        self.skip_drop = skip_drop
        self.one_drop = one_drop
        self.random_state = random_state

    def _validate(self) -> None:
        if int(self.n_estimators) != self.n_estimators or self.n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer")
        if not self.learning_rate > 0:
            raise ValueError("learning_rate must be > 0")
        if not 0.0 <= self.rate_drop <= 1.0:
            raise ValueError("rate_drop must lie in [0, 1]")
        if not 0.0 <= self.skip_drop <= 1.0:
            raise ValueError("skip_drop must lie in [0, 1]")
        if not 0.0 < self.subsample <= 1.0:
            raise ValueError("subsample must lie in (0, 1]")

    def _rngs(self):
        # Tree seeds come from the same stream as GradientBoosting* so that
        # rate_drop=0 reproduces it exactly; dropout uses its own stream.
        tree_rng = np.random.default_rng(self.random_state)
        drop_seed = None if self.random_state is None else [int(self.random_state), 0xDA27]
        return tree_rng, np.random.default_rng(drop_seed)

    def _subsample_idx(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.subsample >= 1.0:
            return np.arange(n)
        return rng.choice(n, max(1, int(self.subsample * n)), replace=False)

    def _select_dropped(self, n_stages: int, rng: np.random.Generator) -> np.ndarray:
        if n_stages == 0 or self.rate_drop <= 0.0:
            return np.empty(0, dtype=np.intp)
        if self.skip_drop > 0.0 and rng.random() < self.skip_drop:
            return np.empty(0, dtype=np.intp)
        mask = rng.random(n_stages) < self.rate_drop
        if self.one_drop and not mask.any():
            mask[rng.integers(n_stages)] = True
        return np.flatnonzero(mask)

    def _new_tree(self, rng: np.random.Generator) -> DecisionTree:
        return DecisionTree(
            criterion="variance",
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_features=self.max_features,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )

    def _normalise(self, weights: List[float], dropped: np.ndarray) -> float:
        k = dropped.size
        eta = self.learning_rate
        if k == 0:
            return eta
        factor = k / (k + eta)
        for j in dropped:
            weights[j] *= factor
        return eta / (k + eta)


class DARTRegressor(_BaseDART):
    """Squared-error gradient boosting with DART dropout.

    Parameters
    ----------
    n_estimators, learning_rate, max_depth, min_samples_split, subsample,
    max_features :
        As for :class:`GradientBoostingRegressor`.
    rate_drop :
        Probability of dropping each existing tree at a boosting stage.
    skip_drop :
        Probability of skipping dropout entirely for a stage.
    one_drop :
        Force at least one dropped tree whenever dropout is applied.
    random_state :
        Seed for reproducibility.
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, min_samples_split: int = 2,
                 subsample: float = 1.0,
                 max_features: Optional[Union[int, str]] = None,
                 rate_drop: float = 0.1, skip_drop: float = 0.0,
                 one_drop: bool = False,
                 random_state: Optional[int] = None) -> None:
        super().__init__(n_estimators, learning_rate, max_depth, min_samples_split,
                         subsample, max_features, rate_drop, skip_drop, one_drop,
                         random_state)

    def fit(self, X, y) -> "DARTRegressor":
        self._validate()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y, dtype=np.float64)
        n = X.shape[0]
        tree_rng, drop_rng = self._rngs()
        self._init = float(np.mean(y))
        self.estimators_: List[DecisionTree] = []
        weights: List[float] = []
        raw: List[np.ndarray] = []  # cached training-set predictions per tree
        self.n_dropped_: List[int] = []
        for _ in range(self.n_estimators):
            dropped = self._select_dropped(len(self.estimators_), drop_rng)
            F = np.full(n, self._init)
            for j, (w, p) in enumerate(zip(weights, raw)):
                F += w * p
            if dropped.size:
                F_drop = F - sum(weights[j] * raw[j] for j in dropped)
            else:
                F_drop = F
            residual = y - F_drop
            idx = self._subsample_idx(n, tree_rng)
            tree = self._new_tree(tree_rng)
            tree.fit(X[idx], residual[idx])
            w_new = self._normalise(weights, dropped)
            self.estimators_.append(tree)
            weights.append(w_new)
            raw.append(tree.predict(X))
            self.n_dropped_.append(int(dropped.size))
        self.tree_weights_ = np.asarray(weights)
        return self

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        F = np.full(X.shape[0], self._init)
        for w, tree in zip(self.tree_weights_, self.estimators_):
            F += w * tree.predict(X)
        return F


class DARTClassifier(_BaseDART):
    """Multinomial-deviance gradient boosting with DART dropout.

    One regression tree per class per stage, as in
    :class:`GradientBoostingClassifier`; dropout acts on whole stages.
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, min_samples_split: int = 2,
                 subsample: float = 1.0,
                 max_features: Optional[Union[int, str]] = None,
                 rate_drop: float = 0.1, skip_drop: float = 0.0,
                 one_drop: bool = False,
                 random_state: Optional[int] = None) -> None:
        super().__init__(n_estimators, learning_rate, max_depth, min_samples_split,
                         subsample, max_features, rate_drop, skip_drop, one_drop,
                         random_state)

    def fit(self, X, y) -> "DARTClassifier":
        self._validate()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        lookup = {c: i for i, c in enumerate(self.classes_)}
        y_int = np.array([lookup[v] for v in y], dtype=np.intp)
        n, K = X.shape[0], self.n_classes_
        tree_rng, drop_rng = self._rngs()
        counts = np.bincount(y_int, minlength=K)
        self._log_prior = np.log(counts / n + 1e-12)
        onehot = np.eye(K)[y_int]
        self.estimators_: List[List[DecisionTree]] = []
        weights: List[float] = []
        raw: List[np.ndarray] = []  # (n, K) per stage
        self.n_dropped_: List[int] = []
        for _ in range(self.n_estimators):
            dropped = self._select_dropped(len(self.estimators_), drop_rng)
            F = np.tile(self._log_prior, (n, 1))
            for w, p in zip(weights, raw):
                F += w * p
            if dropped.size:
                F = F - sum(weights[j] * raw[j] for j in dropped)
            neg_grad = onehot - _softmax(F)
            stage: List[DecisionTree] = []
            stage_raw = np.empty((n, K))
            for k in range(K):
                idx = self._subsample_idx(n, tree_rng)
                tree = self._new_tree(tree_rng)
                tree.fit(X[idx], neg_grad[idx, k])
                stage.append(tree)
                stage_raw[:, k] = tree.predict(X)
            w_new = self._normalise(weights, dropped)
            self.estimators_.append(stage)
            weights.append(w_new)
            raw.append(stage_raw)
            self.n_dropped_.append(int(dropped.size))
        self.tree_weights_ = np.asarray(weights)
        return self

    def _decision(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        F = np.tile(self._log_prior, (X.shape[0], 1))
        for w, stage in zip(self.tree_weights_, self.estimators_):
            for k, tree in enumerate(stage):
                F[:, k] += w * tree.predict(X)
        return F

    def predict_proba(self, X) -> np.ndarray:
        return _softmax(self._decision(X))

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
