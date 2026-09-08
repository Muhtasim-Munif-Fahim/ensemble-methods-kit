"""Bootstrap aggregating (bagging) of decision trees."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .utils import DecisionTree

__all__ = ["BaggingClassifier"]


class BaggingClassifier:
    """A bagging ensemble of :class:`~ensemble_methods_kit.utils.DecisionTree`.

    Each estimator is fit on a bootstrap sample of the training data.
    Predictions are averaged over the per-tree class probabilities (soft
    voting), which is the standard bagging aggregation.

    Parameters
    ----------
    base_estimator :
        A :class:`DecisionTree` used as the template.  ``None`` defaults to a
        deep tree.  Only the constructor parameters are cloned; the random
        state is reset for every tree.
    n_estimators :
        Number of trees in the ensemble.
    max_samples :
        Fraction of the training set sampled (with replacement) per tree.
    max_features :
        Fraction of features considered per tree.  ``None``/``1.0`` uses all
        features.
    bootstrap :
        Whether to use bootstrap samples.  When ``False`` each tree sees the
        full dataset.
    random_state :
        Seed for reproducible bootstrap sampling.
    """

    def __init__(
        self,
        base_estimator: Optional[DecisionTree] = None,
        n_estimators: int = 10,
        max_samples: float = 1.0,
        max_features: float = 1.0,
        bootstrap: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.estimators_: list = []
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: Optional[int] = None

    # ----------------------------------------------------------- cloning
    def _clone_tree(self, seed: int) -> DecisionTree:
        base = self.base_estimator
        common = dict(
            criterion="gini",
            max_depth=5,
            min_samples_split=2,
            min_impurity_decrease=0.0,
            max_features="sqrt",
            random_state=seed,
        )
        if isinstance(base, DecisionTree):
            common = dict(
                criterion=base.criterion,
                max_depth=base.max_depth,
                min_samples_split=base.min_samples_split,
                min_impurity_decrease=base.min_impurity_decrease,
                max_features=base.max_features,
                random_state=seed,
            )
        n_features = getattr(self, "_n_features_in_", None)
        if self.max_features is not None and 0.0 < self.max_features < 1.0 and n_features is not None:
            k = max(1, int(self.max_features * n_features))
            common["max_features"] = k
        return DecisionTree(**common)

    # ---------------------------------------------------------------- fit
    def fit(self, X, y) -> "BaggingClassifier":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]
        self._n_features_in_ = X.shape[1]
        self.estimators_ = []
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        n_sub = int(round(n_samples * self.max_samples))
        n_sub = min(max(n_sub, 1), n_samples)
        for _ in range(self.n_estimators):
            if self.bootstrap:
                idx = rng.integers(0, n_samples, size=n_sub)
            else:
                idx = np.arange(n_samples)
            Xb, yb = X[idx], y[idx]
            tree = self._clone_tree(seed=int(rng.integers(0, 2**31 - 1)))
            tree.fit(Xb, yb)
            self.estimators_.append(tree)
        return self

    # ----------------------------------------------------------- predict
    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        proba = np.mean([tree.predict_proba(X) for tree in self.estimators_], axis=0)
        return proba

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        preds = proba.argmax(axis=1)
        return self.classes_[preds]

    @property
    def oob_score_(self) -> float:
        """Convenience not implemented; use the demo for evaluation."""
        raise AttributeError("oob_score_ is not supported")
