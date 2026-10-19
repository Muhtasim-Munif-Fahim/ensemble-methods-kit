"""Quantile regression forests (Meinshausen, 2006).

A random forest regressor averages leaf *means*, so it only estimates
``E[Y | X = x]``.  Meinshausen's observation is that the same trees define a
data-adaptive nearest-neighbour weighting of the training targets:

    w_i(x) = (1 / T) * sum_t  1{leaf_t(X_i) = leaf_t(x)} / |leaf_t(x)|,

which sums to one, and

    F(y | x) = sum_i w_i(x) 1{Y_i <= y}

is a consistent estimate of the full conditional distribution.  Quantiles of
``F`` give prediction intervals that adapt to heteroscedastic noise without
any distributional assumption, from a single fitted forest.

``QuantileRegressionForest`` subclasses :class:`RandomForestRegressor` (so the
usual ``predict`` still returns the conditional mean) and adds
``predict_quantiles``, ``predict_interval``, ``conditional_weights`` and
out-of-bag versions for honest calibration checks.  ``pinball_loss`` and
``interval_coverage`` evaluate the results.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from .random_forest import RandomForestRegressor
from .utils import r2_score

__all__ = [
    "QuantileRegressionForest",
    "pinball_loss",
    "interval_coverage",
]


def pinball_loss(y_true, y_pred, quantile: float) -> float:
    """Mean pinball (quantile) loss ``mean(max(q (y - p), (q - 1)(y - p)))``."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be in (0, 1)")
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length")
    diff = y_true - y_pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1.0) * diff)))


def interval_coverage(y_true, lower, upper) -> float:
    """Fraction of targets with ``lower <= y <= upper``."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    lower = np.asarray(lower, dtype=np.float64).ravel()
    upper = np.asarray(upper, dtype=np.float64).ravel()
    if not (y_true.shape == lower.shape == upper.shape):
        raise ValueError("y_true, lower and upper must have the same length")
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def _as_quantiles(quantiles) -> Tuple[np.ndarray, bool]:
    scalar = np.ndim(quantiles) == 0
    q = np.atleast_1d(np.asarray(quantiles, dtype=np.float64))
    if q.ndim != 1 or q.size == 0:
        raise ValueError("quantiles must be a float or a 1-d sequence")
    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("quantiles must lie in [0, 1]")
    return q, scalar


class QuantileRegressionForest(RandomForestRegressor):
    """Random forest that estimates conditional quantiles (Meinshausen 2006).

    Parameters are those of :class:`RandomForestRegressor`, plus

    leaf_weighting :
        ``"original"`` (Meinshausen): every training row in a leaf counts
        once, whether or not it was in that tree's bootstrap sample.
        ``"inbag"``: rows count with their bootstrap multiplicity in that
        tree (rows a tree never saw get zero weight there), matching the
        leaf means the tree was grown on.
    chunk_size :
        Query rows processed at once when building the weight matrix.

    Defaults differ from :class:`RandomForestRegressor` in growing deep
    trees (``max_depth=None``, ``min_samples_split=10``): the quantiles come
    from the training targets that share a leaf, so small leaves give
    sharper, more local conditional distributions.

    After :meth:`fit`, ``y_train_`` holds the training targets and
    ``train_leaves_`` the ``(n_estimators, n_samples)`` leaf indices.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 10,
        min_impurity_decrease: float = 0.0,
        max_features: Union[int, str] = "sqrt",
        max_samples: float = 1.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        leaf_weighting: str = "original",
        chunk_size: int = 512,
        random_state: Optional[int] = None,
    ) -> None:
        if leaf_weighting not in ("original", "inbag"):
            raise ValueError('leaf_weighting must be "original" or "inbag"')
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_impurity_decrease=min_impurity_decrease,
            max_features=max_features,
            max_samples=max_samples,
            bootstrap=bootstrap,
            oob_score=oob_score,
            random_state=random_state,
        )
        self.leaf_weighting = leaf_weighting
        self.chunk_size = chunk_size
        self.y_train_: Optional[np.ndarray] = None
        self.train_leaves_: Optional[np.ndarray] = None
        self._inbag_counts: Optional[np.ndarray] = None
        self._leaf_totals: list = []
        self._order: Optional[np.ndarray] = None
        self._train_X: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ fit
    def fit(self, X, y) -> "QuantileRegressionForest":
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
        n = X.shape[0]
        n_sub = min(max(int(round(n * self.max_samples)), 1), n)
        counts = np.zeros((self.n_estimators, n), dtype=np.float64)
        leaves = np.zeros((self.n_estimators, n), dtype=np.intp)
        for t in range(self.n_estimators):
            if self.bootstrap:
                idx = rng.integers(0, n, size=n_sub)
            else:
                idx = np.arange(n)
            tree = self._clone_tree(seed=int(rng.integers(0, 2**31 - 1)))
            tree.fit(X[idx], y[idx])
            self.estimators_.append(tree)
            counts[t] = np.bincount(idx, minlength=n)
            leaves[t] = tree.apply(X)

        self.y_train_ = y
        self._train_X = X
        self.train_leaves_ = leaves
        self._inbag_counts = counts
        self._order = np.argsort(y, kind="stable")
        member = self._membership_weights()
        self._leaf_totals = [
            np.bincount(leaves[t], weights=member[t], minlength=self.estimators_[t].n_leaves_)
            for t in range(self.n_estimators)
        ]
        if self.oob_score:
            pred = self.oob_predict_quantiles(None)
            valid = ~np.isnan(pred)
            if not np.any(valid):
                raise ValueError(
                    "oob_score=True but no out-of-bag samples were produced; increase n_estimators"
                )
            self._oob_prediction = pred
            self._oob_score_value = r2_score(y[valid], pred[valid])
        return self

    def _membership_weights(self) -> np.ndarray:
        if self.leaf_weighting == "inbag":
            return self._inbag_counts
        return np.ones_like(self._inbag_counts)

    def _check_fitted(self) -> None:
        if self.y_train_ is None:
            raise RuntimeError("Estimator is not fitted yet")

    # -------------------------------------------------------------- weights
    def _weights(self, X: np.ndarray, oob_rows: Optional[np.ndarray] = None) -> np.ndarray:
        """Meinshausen weights of the training rows for each query row.

        With ``oob_rows`` (training indices aligned with ``X``) each query row
        only uses trees for which it was out of bag, and its own target gets
        zero weight.
        """
        member = self._membership_weights()
        n_train = self.y_train_.shape[0]
        W = np.zeros((X.shape[0], n_train), dtype=np.float64)
        used = np.zeros(X.shape[0], dtype=np.float64)
        for t, tree in enumerate(self.estimators_):
            q_leaves = tree.apply(X)
            totals = self._leaf_totals[t][q_leaves]
            same = q_leaves[:, None] == self.train_leaves_[t][None, :]
            contrib = same * member[t][None, :]
            if oob_rows is not None:
                active = self._inbag_counts[t][oob_rows] == 0
                contrib[np.arange(X.shape[0]), oob_rows] = 0.0
                totals = contrib.sum(axis=1)
                contrib[~active] = 0.0
            else:
                active = np.ones(X.shape[0], dtype=bool)
            ok = active & (totals > 0)
            W[ok] += contrib[ok] / totals[ok, None]
            used += ok
        nonzero = used > 0
        W[nonzero] /= used[nonzero, None]
        W[~nonzero] = np.nan
        return W

    def conditional_weights(self, X) -> np.ndarray:
        """``(n_query, n_train)`` weights ``w_i(x)``; each row sums to one."""
        self._check_fitted()
        X = self._as_query(X)
        return np.vstack([self._weights(X[i : i + self.chunk_size]) for i in range(0, X.shape[0], self.chunk_size)])

    def _as_query(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self._n_features_in_:
            raise ValueError("X has a different number of features than during fit")
        return X

    def _quantiles_from_weights(self, W: np.ndarray, q: Optional[np.ndarray]) -> np.ndarray:
        y_sorted = self.y_train_[self._order]
        if q is None:
            return W @ self.y_train_
        cdf = np.cumsum(W[:, self._order], axis=1)
        out = np.empty((W.shape[0], q.size), dtype=np.float64)
        for j, alpha in enumerate(q):
            # Smallest y with F(y | x) >= alpha (generalised inverse).
            # For alpha = 0 that is the smallest target with positive weight.
            threshold = alpha - 1e-12 if alpha > 0 else 1e-15
            idx = np.sum(cdf < threshold, axis=1)
            idx = np.minimum(idx, y_sorted.size - 1)
            out[:, j] = y_sorted[idx]
        out[np.isnan(W).any(axis=1)] = np.nan
        return out

    # ------------------------------------------------------------- predict
    def predict_quantiles(self, X, quantiles: Union[float, Sequence[float]] = (0.05, 0.5, 0.95)) -> np.ndarray:
        """Conditional quantiles; shape ``(n, len(quantiles))`` or ``(n,)`` for a float."""
        self._check_fitted()
        q, scalar = _as_quantiles(quantiles)
        X = self._as_query(X)
        parts = [
            self._quantiles_from_weights(self._weights(X[i : i + self.chunk_size]), q)
            for i in range(0, X.shape[0], self.chunk_size)
        ]
        out = np.vstack(parts)
        return out[:, 0] if scalar else out

    def predict_interval(self, X, coverage: float = 0.9) -> Tuple[np.ndarray, np.ndarray]:
        """Central prediction interval with nominal ``coverage``."""
        if not 0.0 < coverage < 1.0:
            raise ValueError("coverage must be in (0, 1)")
        alpha = (1.0 - coverage) / 2.0
        bounds = self.predict_quantiles(X, [alpha, 1.0 - alpha])
        return bounds[:, 0], bounds[:, 1]

    def oob_predict_quantiles(self, quantiles: Union[None, float, Sequence[float]] = (0.05, 0.5, 0.95)) -> np.ndarray:
        """Out-of-bag conditional quantiles for every training row.

        Row ``i`` uses only trees whose bootstrap sample left it out and
        excludes its own target, so these are honest held-out predictions
        (``nan`` for rows that were never out of bag).  ``quantiles=None``
        returns the OOB weighted conditional mean.
        """
        self._check_fitted()
        if not self.bootstrap:
            raise ValueError("out-of-bag predictions require bootstrap=True")
        X_rows = np.arange(self.y_train_.shape[0])
        q = None
        scalar = False
        if quantiles is not None:
            q, scalar = _as_quantiles(quantiles)
        Xtr = self._train_X
        parts = []
        for i in range(0, X_rows.size, self.chunk_size):
            rows = X_rows[i : i + self.chunk_size]
            W = self._weights(Xtr[rows], oob_rows=rows)
            if q is None:
                vals = np.where(np.isnan(W).any(axis=1), np.nan, np.nan_to_num(W) @ self.y_train_)
                parts.append(vals[:, None])
            else:
                parts.append(self._quantiles_from_weights(W, q))
        out = np.vstack(parts)
        if q is None or scalar:
            return out[:, 0]
        return out
