"""Bootstrap aggregating (bagging) of decision trees."""

from __future__ import annotations

from typing import Optional

import numpy as np

from .utils import DecisionTree, mean_decrease_impurity, r2_score

__all__ = ["BaggingClassifier", "BaggingRegressor"]


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
            splitter="best",
            random_state=seed,
        )
        if isinstance(base, DecisionTree):
            common = dict(
                criterion=base.criterion,
                max_depth=base.max_depth,
                min_samples_split=base.min_samples_split,
                min_impurity_decrease=base.min_impurity_decrease,
                max_features=base.max_features,
                splitter=getattr(base, "splitter", "best"),
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

    @property
    def feature_importances_(self) -> np.ndarray:
        """Normalized Mean Decrease Impurity (MDI) importances.

        Averages the impurity-decrease importances of the fitted trees and
        renormalizes so the vector sums to 1.  Inherited by
        :class:`~ensemble_methods_kit.random_forest.RandomForestClassifier`
        and :class:`~ensemble_methods_kit.extra_trees.ExtraTreesClassifier`.
        """
        if not self.estimators_:
            raise RuntimeError("Estimator is not fitted yet")
        return mean_decrease_impurity(self.estimators_)


class BaggingRegressor:
    """A bagging ensemble of regression trees.

    Each estimator is a :class:`~ensemble_methods_kit.utils.DecisionTree`
    grown with squared-error (``criterion="variance"``) on a bootstrap sample
    of the training rows.  Predictions are the mean of the per-tree outputs,
    which is the standard bagging aggregation for regression.

    Parameters
    ----------
    base_estimator :
        A :class:`DecisionTree` used as the template for depth and split
        controls.  ``None`` defaults to a depth-5 tree that sees every
        feature.  The criterion is always variance; only the structural
        hyperparameters are cloned, and the random state is reset per tree.
    n_estimators :
        Number of trees in the ensemble.
    max_samples :
        Fraction of the training set drawn for each tree.  With
        ``bootstrap=True`` the draw is with replacement.
    max_features :
        Fraction of features considered per tree.  ``None`` or ``1.0`` uses
        every feature.  A value in ``(0, 1)`` is converted to an integer
        count and forwarded to each tree's ``max_features``.
    bootstrap :
        Whether to use bootstrap samples.  When ``False`` each tree sees the
        full dataset (and out-of-bag scoring is unavailable).
    oob_score :
        When ``True`` and ``bootstrap=True``, compute the out-of-bag R² on
        training rows that were left out of at least one tree.  The score is
        stored on :attr:`oob_score_` and the averaged left-out predictions on
        :attr:`oob_prediction_`.
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
        oob_score: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        if not 0.0 < float(max_samples) <= 1.0:
            raise ValueError("max_samples must be in (0, 1]")
        if max_features is not None and not 0.0 < float(max_features) <= 1.0:
            raise ValueError("max_features must be in (0, 1] or None")
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        self.estimators_: list = []
        self._n_features_in_: Optional[int] = None
        self._oob_prediction: Optional[np.ndarray] = None
        self._oob_score_value: Optional[float] = None

    def _clone_tree(self, seed: int) -> DecisionTree:
        base = self.base_estimator
        common = dict(
            criterion="variance",
            max_depth=5,
            min_samples_split=2,
            min_impurity_decrease=0.0,
            max_features=None,
            splitter="best",
            random_state=seed,
        )
        if isinstance(base, DecisionTree):
            common.update(
                max_depth=base.max_depth,
                min_samples_split=base.min_samples_split,
                min_impurity_decrease=base.min_impurity_decrease,
                max_features=base.max_features,
                splitter=getattr(base, "splitter", "best"),
            )
        n_features = getattr(self, "_n_features_in_", None)
        if (
            self.max_features is not None
            and 0.0 < float(self.max_features) < 1.0
            and n_features is not None
        ):
            k = max(1, int(float(self.max_features) * n_features))
            common["max_features"] = k
        return DecisionTree(**common)

    def fit(self, X, y) -> "BaggingRegressor":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        if y.ndim != 1:
            raise ValueError("y must be a 1-d target vector")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        if self.oob_score and not self.bootstrap:
            raise ValueError("oob_score requires bootstrap=True")

        self._n_features_in_ = X.shape[1]
        self.estimators_ = []
        self._oob_prediction = None
        self._oob_score_value = None
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        n_sub = int(round(n_samples * self.max_samples))
        n_sub = min(max(n_sub, 1), n_samples)
        if self.oob_score:
            oob_sum = np.zeros(n_samples, dtype=np.float64)
            oob_count = np.zeros(n_samples, dtype=np.float64)

        for _ in range(self.n_estimators):
            if self.bootstrap:
                idx = rng.integers(0, n_samples, size=n_sub)
            else:
                idx = np.arange(n_samples)
            tree = self._clone_tree(seed=int(rng.integers(0, 2**31 - 1)))
            tree.fit(X[idx], y[idx])
            self.estimators_.append(tree)
            if self.oob_score:
                in_bag = np.zeros(n_samples, dtype=bool)
                in_bag[idx] = True
                oob_mask = ~in_bag
                if np.any(oob_mask):
                    oob_sum[oob_mask] += tree.predict(X[oob_mask])
                    oob_count[oob_mask] += 1.0

        if self.oob_score:
            valid = oob_count > 0
            if not np.any(valid):
                raise ValueError(
                    "oob_score=True but no out-of-bag samples were produced; "
                    "increase n_estimators"
                )
            pred = np.full(n_samples, np.nan, dtype=np.float64)
            pred[valid] = oob_sum[valid] / oob_count[valid]
            self._oob_prediction = pred
            self._oob_score_value = r2_score(y[valid], pred[valid])
        return self

    def predict(self, X) -> np.ndarray:
        if not self.estimators_:
            raise RuntimeError("Estimator is not fitted yet")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        preds = [tree.predict(X) for tree in self.estimators_]
        return np.mean(preds, axis=0)

    @property
    def oob_prediction_(self) -> np.ndarray:
        """Out-of-bag predictions, one per training row.

        Available only after :meth:`fit` with ``oob_score=True``.  Rows that
        were never left out are ``nan``.
        """
        if not self.estimators_:
            raise RuntimeError("Estimator is not fitted yet")
        if self._oob_prediction is None:
            raise AttributeError("oob_prediction_ is only available when oob_score=True")
        return self._oob_prediction

    @property
    def oob_score_(self) -> float:
        """Out-of-bag coefficient of determination (R²).

        Available only after :meth:`fit` with ``oob_score=True``.  The score
        uses training rows that were left out of at least one bootstrap
        sample.
        """
        if not self.estimators_:
            raise RuntimeError("Estimator is not fitted yet")
        if self._oob_score_value is None:
            raise AttributeError("oob_score_ is only available when oob_score=True")
        return self._oob_score_value

    @property
    def feature_importances_(self) -> np.ndarray:
        """Normalized Mean Decrease Impurity (MDI) importances.

        Averages the impurity-decrease importances of the fitted regression
        trees and renormalizes so the vector sums to 1.  Inherited by
        :class:`~ensemble_methods_kit.random_forest.RandomForestRegressor`.
        """
        if not self.estimators_:
            raise RuntimeError("Estimator is not fitted yet")
        return mean_decrease_impurity(self.estimators_)
