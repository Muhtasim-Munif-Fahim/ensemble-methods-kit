"""Histogram-based gradient boosting classifier.

``GradientBoostingClassifier`` grows exact CART trees on the raw features.
This estimator first maps every feature onto ordered bins (quantile edges,
or the observed distinct values when there are fewer of them than
``max_bins``) and then grows Newton trees by scanning those bins.  Binning
is the histogram trick popularised by LightGBM (Ke et al., 2017) and used
by scikit-learn's ``HistGradientBoostingClassifier``.

Binary problems use one tree per iteration and a log-loss / sigmoid link.
Multiclass problems use one tree per class per iteration and a diagonal
Newton step on the softmax cross-entropy.  ``staged_predict_proba`` yields
the class probabilities after every boosting iteration.
"""

from __future__ import annotations

from typing import Iterator, List, Optional

import numpy as np

__all__ = ["HistogramGradientBoostingClassifier"]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty(z.shape, dtype=np.float64)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def _bin_thresholds(column: np.ndarray, max_bins: int) -> np.ndarray:
    """Return interior cut points so ``searchsorted(..., side='right')`` bins ``column``.

    Constant columns produce no cuts (every value shares one bin).  When the
    column has at most ``max_bins`` distinct values, each value gets its own
    bin.  Otherwise the cuts are unique interior quantiles, giving at most
    ``max_bins`` bins.
    """
    distinct = np.unique(column)
    if distinct.size <= 1:
        return np.empty(0, dtype=np.float64)
    if distinct.size <= max_bins:
        return np.ascontiguousarray(distinct[1:], dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, int(max_bins) + 1)[1:-1]
    thresholds = np.unique(np.quantile(column, quantiles))
    return np.ascontiguousarray(thresholds, dtype=np.float64)


def _bin_matrix(X: np.ndarray, thresholds: List[np.ndarray]) -> np.ndarray:
    binned = np.empty(X.shape, dtype=np.int32)
    for j, edges in enumerate(thresholds):
        binned[:, j] = np.searchsorted(edges, X[:, j], side="right")
    return binned


class _HistNode:
    """A node in a histogram tree.

    Leaves store the Newton step in ``value``.  Internal nodes send rows
    with ``binned[:, feature] <= threshold`` to the left child.
    """

    __slots__ = ("value", "feature", "threshold", "left", "right")

    def __init__(self, value: float) -> None:
        self.value = float(value)
        self.feature: Optional[int] = None
        self.threshold = 0
        self.left: Optional[_HistNode] = None
        self.right: Optional[_HistNode] = None


def _apply_tree(node: _HistNode, binned: np.ndarray) -> np.ndarray:
    """Evaluate a histogram tree on an already-binned matrix."""
    out = np.empty(binned.shape[0], dtype=np.float64)
    for i in range(binned.shape[0]):
        current = node
        row = binned[i]
        while current.feature is not None:
            if row[current.feature] <= current.threshold:
                child = current.left
            else:
                child = current.right
            if child is None:
                break
            current = child
        out[i] = current.value
    return out


class HistogramGradientBoostingClassifier:
    """Histogram gradient boosting for binary and multiclass classification.

    Each boosting iteration fits one or more regression trees to the Newton
    step of the loss (log-loss for two classes, multinomial deviance
    otherwise).  Split candidates are the bins of each feature rather than
    every mid-point of the raw column, which is what makes the method a
    histogram booster.

    After :meth:`fit`, :attr:`feature_importances_` is the total split gain
    of each feature, normalised to sum to 1.  Features that are constant,
    or that the trees never split on, receive 0.  This is gain importance,
    not the Mean Decrease Impurity reported by the CART ensembles.

    Parameters
    ----------
    n_estimators :
        Number of boosting iterations.  Binary problems grow one tree per
        iteration; multiclass problems grow one tree per class per iteration
        (scikit-learn's ``max_iter``).
    learning_rate :
        Shrinkage applied to each tree's Newton step.
    max_depth :
        Maximum depth of each tree.  ``1`` yields a single split.
    max_bins :
        Maximum number of bins per feature.  Columns with fewer distinct
        values use one bin per value.
    min_samples_leaf :
        Minimum number of samples required in each child to accept a split.
    l2_regularization :
        L2 penalty added to the summed Hessian in the leaf value and the
        split gain.  ``0`` recovers an unpenalised Newton step.
    subsample :
        Fraction of training rows used to grow each tree.  ``1.0`` uses
        every row.  Scores are still updated on the full training set.
    random_state :
        Seed for row sub-sampling when ``subsample < 1``.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        max_bins: int = 255,
        min_samples_leaf: int = 2,
        l2_regularization: float = 0.0,
        subsample: float = 1.0,
        random_state: Optional[int] = None,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if max_bins < 2:
            raise ValueError("max_bins must be at least 2")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1")
        if l2_regularization < 0:
            raise ValueError("l2_regularization must be non-negative")
        if not 0.0 < subsample <= 1.0:
            raise ValueError("subsample must be in (0, 1]")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_bins = max_bins
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.subsample = subsample
        self.random_state = random_state
        self.estimators_: List[List[_HistNode]] = []
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: Optional[int] = None
        self.bin_thresholds_: List[np.ndarray] = []
        self.train_score_: List[float] = []
        self.n_iter_: int = 0
        self._baseline: Optional[np.ndarray] = None
        self._n_features: Optional[int] = None
        self._gain_sum: Optional[np.ndarray] = None

    def fit(self, X, y) -> "HistogramGradientBoostingClassifier":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X must contain at least one sample and one feature")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        if y.ndim != 1:
            raise ValueError("y must be a 1-d array of class labels")
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        self.classes_, y_int = np.unique(y, return_inverse=True)
        n_classes = int(self.classes_.shape[0])
        if n_classes < 2:
            raise ValueError(
                "HistogramGradientBoostingClassifier requires at least 2 classes"
            )
        self.n_classes_ = n_classes
        n_samples, n_features = X.shape
        self._n_features = n_features
        self.bin_thresholds_ = [
            _bin_thresholds(X[:, j], self.max_bins) for j in range(n_features)
        ]
        binned = _bin_matrix(X, self.bin_thresholds_)
        rng = np.random.default_rng(self.random_state)
        self._gain_sum = np.zeros(n_features, dtype=np.float64)
        self.estimators_ = []
        self.train_score_ = []

        if n_classes == 2:
            self._fit_binary(y_int, binned, rng)
        else:
            self._fit_multiclass(y_int, binned, rng)
        self.n_iter_ = len(self.estimators_)
        return self

    def _row_subset(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        if self.subsample >= 1.0:
            return np.arange(n_samples, dtype=np.intp)
        n_sub = max(1, int(round(self.subsample * n_samples)))
        n_sub = min(n_samples, n_sub)
        return rng.choice(n_samples, size=n_sub, replace=False)

    def _leaf_value(self, grad: np.ndarray, hess: np.ndarray, indices: np.ndarray) -> float:
        summed_grad = float(grad[indices].sum())
        summed_hess = float(hess[indices].sum())
        denom = summed_hess + self.l2_regularization
        if denom <= 1e-12:
            return 0.0
        return float(-summed_grad / denom)

    def _best_split(self, binned, grad, hess, indices):
        n_node = int(indices.shape[0])
        if n_node < 2 * self.min_samples_leaf:
            return None
        summed_grad = float(grad[indices].sum())
        summed_hess = float(hess[indices].sum())
        denom_parent = summed_hess + self.l2_regularization
        if denom_parent <= 1e-12:
            return None
        parent = (summed_grad * summed_grad) / denom_parent
        best_gain = 0.0
        best = None
        min_leaf = self.min_samples_leaf
        l2 = self.l2_regularization

        for feature in range(binned.shape[1]):
            bins_f = binned[indices, feature]
            if int(bins_f.min()) == int(bins_f.max()):
                continue
            order = np.argsort(bins_f, kind="mergesort")
            sorted_bins = bins_f[order]
            sorted_idx = indices[order]
            grad_cum = np.cumsum(grad[sorted_idx])
            hess_cum = np.cumsum(hess[sorted_idx])
            boundaries = np.flatnonzero(sorted_bins[:-1] != sorted_bins[1:])
            if boundaries.size == 0:
                continue
            n_left = boundaries + 1
            n_right = n_node - n_left
            eligible = (n_left >= min_leaf) & (n_right >= min_leaf)
            if not np.any(eligible):
                continue
            boundaries = boundaries[eligible]
            grad_left = grad_cum[boundaries]
            hess_left = hess_cum[boundaries]
            grad_right = summed_grad - grad_left
            hess_right = summed_hess - hess_left
            denom_left = hess_left + l2
            denom_right = hess_right + l2
            stable = (denom_left > 1e-12) & (denom_right > 1e-12)
            if not np.any(stable):
                continue
            boundaries = boundaries[stable]
            grad_left = grad_left[stable]
            grad_right = grad_right[stable]
            denom_left = denom_left[stable]
            denom_right = denom_right[stable]
            gains = 0.5 * (
                (grad_left * grad_left) / denom_left
                + (grad_right * grad_right) / denom_right
                - parent
            )
            local = int(np.argmax(gains))
            gain = float(gains[local])
            if gain > best_gain:
                best_gain = gain
                pos = int(boundaries[local])
                best = (
                    feature,
                    int(sorted_bins[pos]),
                    sorted_idx[: pos + 1].copy(),
                    sorted_idx[pos + 1 :].copy(),
                    gain,
                )
        return best

    def _grow(self, binned, grad, hess, indices, depth: int) -> _HistNode:
        node = _HistNode(self._leaf_value(grad, hess, indices))
        if depth >= self.max_depth:
            return node
        split = self._best_split(binned, grad, hess, indices)
        if split is None:
            return node
        feature, threshold, left_idx, right_idx, gain = split
        node.feature = int(feature)
        node.threshold = int(threshold)
        self._gain_sum[feature] += gain
        node.left = self._grow(binned, grad, hess, left_idx, depth + 1)
        node.right = self._grow(binned, grad, hess, right_idx, depth + 1)
        return node

    def _fit_binary(self, y_int, binned, rng) -> None:
        y_bin = (y_int == 1).astype(np.float64)
        n_samples = y_bin.shape[0]
        positive_rate = float(np.clip(y_bin.mean(), 1e-12, 1.0 - 1e-12))
        baseline = np.log(positive_rate / (1.0 - positive_rate))
        self._baseline = np.array([baseline], dtype=np.float64)
        scores = np.full(n_samples, baseline, dtype=np.float64)
        for _ in range(self.n_estimators):
            probability = _sigmoid(scores)
            gradient = probability - y_bin
            hessian = np.maximum(probability * (1.0 - probability), 1e-6)
            tree = self._grow(
                binned, gradient, hessian, self._row_subset(n_samples, rng), depth=0
            )
            scores = scores + self.learning_rate * _apply_tree(tree, binned)
            self.estimators_.append([tree])
            self.train_score_.append(_binary_log_loss(y_bin, scores))

    def _fit_multiclass(self, y_int, binned, rng) -> None:
        n_samples = y_int.shape[0]
        n_classes = self.n_classes_
        counts = np.bincount(y_int, minlength=n_classes).astype(np.float64)
        priors = np.clip(counts / n_samples, 1e-12, None)
        priors = priors / priors.sum()
        self._baseline = np.log(priors)
        scores = np.tile(self._baseline, (n_samples, 1))
        for _ in range(self.n_estimators):
            probability = _softmax(scores)
            subset = self._row_subset(n_samples, rng)
            stage: List[_HistNode] = []
            for class_index in range(n_classes):
                class_probability = probability[:, class_index]
                gradient = class_probability - (y_int == class_index).astype(np.float64)
                hessian = np.maximum(class_probability * (1.0 - class_probability), 1e-6)
                stage.append(
                    self._grow(binned, gradient, hessian, subset, depth=0)
                )
            for class_index, tree in enumerate(stage):
                scores[:, class_index] += self.learning_rate * _apply_tree(tree, binned)
            self.estimators_.append(stage)
            self.train_score_.append(_multiclass_log_loss(y_int, scores))

    def _check_fitted(self) -> None:
        if not self.estimators_ or self._baseline is None or self.n_classes_ is None:
            raise RuntimeError("Estimator is not fitted yet")

    def _validate_X(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values")
        if self._n_features is not None and X.shape[1] != self._n_features:
            raise ValueError(
                f"X has {X.shape[1]} features, but HistogramGradientBoostingClassifier "
                f"is expecting {self._n_features} features"
            )
        return X

    def _scores_through(self, X) -> Iterator[np.ndarray]:
        """Yield the raw scores after each boosting iteration."""
        self._check_fitted()
        X = self._validate_X(X)
        binned = _bin_matrix(X, self.bin_thresholds_)
        n_samples = binned.shape[0]
        if self.n_classes_ == 2:
            scores = np.full(n_samples, float(self._baseline[0]), dtype=np.float64)
            for stage in self.estimators_:
                scores = scores + self.learning_rate * _apply_tree(stage[0], binned)
                yield scores
        else:
            scores = np.tile(self._baseline, (n_samples, 1))
            for stage in self.estimators_:
                for class_index, tree in enumerate(stage):
                    scores[:, class_index] += self.learning_rate * _apply_tree(tree, binned)
                yield scores.copy()

    @staticmethod
    def _scores_to_proba(scores: np.ndarray, n_classes: int) -> np.ndarray:
        if n_classes == 2:
            positive = _sigmoid(scores)
            return np.column_stack((1.0 - positive, positive))
        return _softmax(scores)

    def staged_predict_proba(self, X) -> Iterator[np.ndarray]:
        """Yield class probabilities after each boosting iteration.

        The generator validates that the estimator is fitted before the
        first probability matrix is produced.  Each matrix has shape
        ``(n_samples, n_classes)`` and rows that sum to one.  The last
        matrix is identical to :meth:`predict_proba`.
        """
        self._check_fitted()
        X = self._validate_X(X)
        n_classes = int(self.n_classes_)
        for scores in self._scores_through(X):
            yield self._scores_to_proba(scores, n_classes)

    def staged_predict(self, X) -> Iterator[np.ndarray]:
        """Yield class labels after each boosting iteration."""
        self._check_fitted()
        X = self._validate_X(X)
        for proba in self.staged_predict_proba(X):
            yield self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X) -> np.ndarray:
        if not self.estimators_:
            raise RuntimeError("Estimator is not fitted yet")
        proba = None
        for proba in self.staged_predict_proba(X):
            pass
        return proba

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    @property
    def feature_importances_(self) -> np.ndarray:
        """Normalised sum of histogram-split gains for each input feature."""
        self._check_fitted()
        if self._gain_sum is None:
            raise RuntimeError("Estimator is not fitted yet")
        importances = np.array(self._gain_sum, dtype=np.float64, copy=True)
        total = float(importances.sum())
        if total > 0.0:
            importances /= total
        return importances


def _binary_log_loss(y_bin: np.ndarray, scores: np.ndarray) -> float:
    probability = np.clip(_sigmoid(scores), 1e-15, 1.0 - 1e-15)
    loss = -(y_bin * np.log(probability) + (1.0 - y_bin) * np.log(1.0 - probability))
    return float(np.mean(loss))


def _multiclass_log_loss(y_int: np.ndarray, scores: np.ndarray) -> float:
    probability = np.clip(_softmax(scores), 1e-15, 1.0)
    return float(-np.mean(np.log(probability[np.arange(y_int.shape[0]), y_int])))
