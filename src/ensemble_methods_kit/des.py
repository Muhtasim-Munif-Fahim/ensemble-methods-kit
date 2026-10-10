"""Dynamic ensemble selection: KNORA-Eliminate and KNORA-Union.

Ko, Sabourin & Britto (2008), "From dynamic classifier selection to dynamic
ensemble selection".  A pool of base classifiers is fitted on a training split.
A held-out *dynamic selection* set (DSEL) is then used at prediction time: for
each query point the ``k`` nearest DSEL neighbours define its *region of
competence*, and only the pool members that are competent in that region vote.

- **KNORA-Eliminate** (``method="eliminate"``) keeps the classifiers that label
  *every* one of the ``k`` neighbours correctly.  When no classifier is that
  good, the neighbourhood shrinks (``k-1``, ``k-2``, ...) until at least one
  oracle is found; if even the single nearest neighbour fools every model, the
  whole pool votes.
- **KNORA-Union** (``method="union"``) lets every classifier vote, weighted by
  the number of neighbours it classifies correctly.

Selection is local, so different regions of the input space can be handled by
different pool members, which is where static averaging tends to lose accuracy.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .utils import DecisionTree, clone_estimator, train_test_split

__all__ = ["KNORAClassifier"]

_METHODS = ("eliminate", "union")


def _as_2d(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
    return X


class KNORAClassifier:
    """KNORA-E / KNORA-U dynamic ensemble selection classifier.

    Parameters
    ----------
    estimators :
        Optional list of ``(name, estimator)`` pairs forming the pool.  Each is
        cloned before fitting.  ``None`` builds a bagged pool of
        ``n_estimators`` :class:`DecisionTree` models on bootstrap samples.
    n_estimators :
        Pool size when ``estimators`` is ``None``.
    max_depth :
        Depth of the default decision-tree pool members.
    method :
        ``"eliminate"`` (KNORA-E) or ``"union"`` (KNORA-U).
    k :
        Size of the region of competence (nearest DSEL neighbours).
    dsel_fraction :
        Fraction of the training data held out as the dynamic selection set.
    standardize :
        Z-score features (using DSEL statistics) before the neighbour search.
    random_state :
        Seed for the DSEL split and the default bootstrap pool.

    Attributes
    ----------
    estimators_ : list
        Fitted pool members.
    classes_ : ndarray
        Class labels.
    dsel_correct_ : ndarray of shape (n_dsel, n_estimators)
        Whether each pool member classifies each DSEL point correctly.
    pool_accuracy_ : ndarray of shape (n_estimators,)
        DSEL accuracy of each pool member.
    """

    def __init__(
        self,
        estimators: Optional[Sequence[Tuple[str, object]]] = None,
        n_estimators: int = 15,
        max_depth: int = 3,
        method: str = "eliminate",
        k: int = 7,
        dsel_fraction: float = 0.3,
        standardize: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        self.estimators = None if estimators is None else list(estimators)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.method = method
        self.k = k
        self.dsel_fraction = dsel_fraction
        self.standardize = standardize
        self.random_state = random_state
        self.estimators_: Optional[List] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
        self.dsel_correct_: Optional[np.ndarray] = None
        self.pool_accuracy_: Optional[np.ndarray] = None
        self._check_params()

    # -- validation -------------------------------------------------------
    def _check_params(self) -> None:
        if self.method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}")
        if not isinstance(self.k, (int, np.integer)) or self.k < 1:
            raise ValueError("k must be a positive integer")
        if not 0.0 < float(self.dsel_fraction) < 1.0:
            raise ValueError("dsel_fraction must be in (0, 1)")
        if self.estimators is None:
            if not isinstance(self.n_estimators, (int, np.integer)) or self.n_estimators < 1:
                raise ValueError("n_estimators must be a positive integer")
        else:
            if len(self.estimators) == 0 or not all(
                isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
                for item in self.estimators
            ):
                raise ValueError("estimators must be a non-empty list of (name, estimator) pairs")
            names = [name for name, _ in self.estimators]
            if len(names) != len(set(names)):
                raise ValueError("estimator names must be unique")
            for name, est in self.estimators:
                if not hasattr(est, "fit") or not hasattr(est, "predict"):
                    raise ValueError(f"estimator {name!r} must implement fit and predict")

    def _check_fitted(self) -> None:
        if self.estimators_ is None:
            raise RuntimeError("Estimator is not fitted yet")

    def _validate_X(self, X) -> np.ndarray:
        X = _as_2d(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {type(self).__name__} is expecting "
                f"{self.n_features_in_} features as input"
            )
        return X

    # -- fitting ----------------------------------------------------------
    def _build_pool(self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> List:
        pool = []
        if self.estimators is None:
            n = X.shape[0]
            for _ in range(int(self.n_estimators)):
                seed = int(rng.integers(0, 2**31 - 1))
                idx = rng.integers(0, n, size=n)
                # guarantee every class appears so predict() spans all labels
                if np.unique(y[idx]).shape[0] < np.unique(y).shape[0]:
                    idx = np.concatenate([idx, [np.flatnonzero(y == c)[0] for c in np.unique(y)]])
                tree = DecisionTree(max_depth=self.max_depth, random_state=seed)
                pool.append(tree.fit(X[idx], y[idx]))
        else:
            for _, est in self.estimators:
                clone = clone_estimator(est)
                clone.fit(X, y)
                pool.append(clone)
        return pool

    def fit(self, X, y) -> "KNORAClassifier":
        self._check_params()
        X = _as_2d(X)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        self.classes_ = np.unique(y)
        if self.classes_.shape[0] < 2:
            raise ValueError("KNORAClassifier needs at least two classes")
        self.n_features_in_ = X.shape[1]
        rng = np.random.default_rng(self.random_state)
        split_seed = int(rng.integers(0, 2**31 - 1))
        X_tr, X_ds, y_tr, y_ds = train_test_split(
            X, y, test_size=float(self.dsel_fraction), random_state=split_seed, stratify=y
        )
        if X_ds.shape[0] < 1 or X_tr.shape[0] < 1:
            raise ValueError("not enough samples to build a training and DSEL split")
        self.estimators_ = self._build_pool(X_tr, y_tr, rng)
        self._dsel_X = X_ds
        if self.standardize:
            self._mu = X_ds.mean(axis=0)
            sd = X_ds.std(axis=0)
            self._sd = np.where(sd > 0, sd, 1.0)
        else:
            self._mu = np.zeros(X.shape[1])
            self._sd = np.ones(X.shape[1])
        self._dsel_Z = (X_ds - self._mu) / self._sd
        preds = np.column_stack([np.asarray(est.predict(X_ds)) for est in self.estimators_])
        self.dsel_correct_ = preds == y_ds[:, None]
        self.pool_accuracy_ = self.dsel_correct_.mean(axis=0)
        return self

    # -- prediction -------------------------------------------------------
    def _neighbours(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self._mu) / self._sd
        d2 = (
            np.sum(Z**2, axis=1)[:, None]
            - 2.0 * Z @ self._dsel_Z.T
            + np.sum(self._dsel_Z**2, axis=1)[None, :]
        )
        k = min(int(self.k), self._dsel_Z.shape[0])
        return np.argsort(d2, axis=1, kind="stable")[:, :k]

    def competence_weights(self, X) -> np.ndarray:
        """Per-query voting weights over the pool, shape ``(n_samples, n_estimators)``.

        For KNORA-E the weights are 0/1 masks of the selected oracles; for
        KNORA-U they are the counts of correctly classified neighbours.
        """
        self._check_fitted()
        X = self._validate_X(X)
        nbrs = self._neighbours(X)
        correct = self.dsel_correct_[nbrs]  # (n, k, m)
        n_models = len(self.estimators_)
        if self.method == "union":
            weights = correct.sum(axis=1).astype(np.float64)
            empty = weights.sum(axis=1) == 0
            weights[empty] = 1.0
            return weights
        weights = np.zeros((X.shape[0], n_models), dtype=np.float64)
        for i in range(X.shape[0]):
            chosen = None
            for size in range(correct.shape[1], 0, -1):
                oracle = np.all(correct[i, :size], axis=0)
                if oracle.any():
                    chosen = oracle
                    break
            weights[i] = 1.0 if chosen is None else chosen.astype(np.float64)
        return weights

    def predict_proba(self, X) -> np.ndarray:
        """Competence-weighted vote shares for each class."""
        self._check_fitted()
        X = self._validate_X(X)
        weights = self.competence_weights(X)
        preds = np.column_stack([np.asarray(est.predict(X)) for est in self.estimators_])
        proba = np.zeros((X.shape[0], self.classes_.shape[0]), dtype=np.float64)
        for c_idx, c in enumerate(self.classes_):
            proba[:, c_idx] = np.sum(weights * (preds == c), axis=1)
        totals = proba.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1.0
        return proba / totals

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def n_selected(self, X) -> np.ndarray:
        """Number of pool members that vote (non-zero weight) for each query."""
        return np.count_nonzero(self.competence_weights(X), axis=1)
