"""Shared utilities: metrics, train/test splitting and the CART decision tree.

The decision tree implemented here is the single workhorse estimator used by
every bagging-style ensemble in the package.  It supports both multi-class
classification (gini or entropy impurity) and squared-error regression.
"""

from __future__ import annotations

import inspect
from typing import List, Optional, Sequence, Union

import numpy as np

__all__ = [
    "train_test_split",
    "bootstrap_sample",
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "mean_squared_error",
    "r2_score",
    "log_loss",
    "clone_estimator",
    "DecisionTree",
]


ArrayLike = Union[Sequence, np.ndarray]


def confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike) -> np.ndarray:
    """Return a 2x2 confusion matrix for binary labels.

    Layout: ``[[tn, fp], [fn, tp]]`` where positive class is the larger
    label value.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true, y_pred]))
    if len(labels) == 1:
        labels = np.array([labels[0], labels[0] + 1])
    pos = labels[-1]
    neg = labels[0]
    tp = float(np.sum((y_true == pos) & (y_pred == pos)))
    tn = float(np.sum((y_true == neg) & (y_pred == neg)))
    fp = float(np.sum((y_true == neg) & (y_pred == pos)))
    fn = float(np.sum((y_true == pos) & (y_pred == neg)))
    return np.array([[tn, fp], [fn, tp]])


def precision_score(y_true: ArrayLike, y_pred: ArrayLike, zero_division: float = 0.0) -> float:
    """Precision = TP / (TP + FP)."""
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    fp = cm[0, 1]
    denom = tp + fp
    return float(tp / denom) if denom > 0 else zero_division


def recall_score(y_true: ArrayLike, y_pred: ArrayLike, zero_division: float = 0.0) -> float:
    """Recall = TP / (TP + FN)."""
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    fn = cm[1, 0]
    denom = tp + fn
    return float(tp / denom) if denom > 0 else zero_division


def f1_score(y_true: ArrayLike, y_pred: ArrayLike, zero_division: float = 0.0) -> float:
    """F1 = 2 * (precision * recall) / (precision + recall)."""
    p = precision_score(y_true, y_pred, zero_division=zero_division)
    r = recall_score(y_true, y_pred, zero_division=zero_division)
    if p + r == 0:
        return zero_division
    return 2.0 * p * r / (p + r)


def bootstrap_sample(
    X: ArrayLike,
    y: ArrayLike,
    *,
    n_samples: int | None = None,
    replace: bool = True,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a bootstrap (resampled with replacement) sample from (X, y).

    Each row is selected independently; with ``replace=True`` some rows appear
    multiple times and others are absent (the OOB set). When ``replace=False``
    the function performs a shuffled subsample without replacement. The row
    pairings between ``X`` and ``y`` are always preserved.

    Parameters
    ----------
    X, y:
        Feature matrix and target vector with matching first-dimension lengths.
    n_samples:
        Number of rows to draw. Defaults to ``len(X)``.
    replace:
        Whether to sample with replacement (bootstrap) or without (subsample).
    random_state:
        Seed for reproducibility.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    n = X.shape[0]
    if n == 0:
        raise ValueError("cannot bootstrap an empty dataset")
    if n_samples is None:
        n_samples = n
    if not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    if not replace and n_samples > n:
        raise ValueError("n_samples cannot exceed the dataset size when replace=False")

    rng = np.random.default_rng(random_state)
    indices = rng.integers(0, n, size=n_samples) if replace else rng.permutation(n)[:n_samples]
    return X[indices], y[indices]


def balanced_sample_weights(y: ArrayLike) -> np.ndarray:
    """Return class-balanced sample weights for imbalanced classification.

    Each sample receives weight ``n_samples / (n_classes * count(class))``
    so that the effective weight of each class sums to the same value. This
    is the standard inverse-frequency balancing used by scikit-learn's
    ``class_weight='balanced'``.

    Parameters
    ----------
    y:
        Target vector of class labels.
    """
    y = np.asarray(y)
    if y.size == 0:
        raise ValueError("cannot compute weights for an empty target vector")
    classes, counts = np.unique(y, return_counts=True)
    if classes.shape[0] < 2:
        raise ValueError("balanced_sample_weights requires at least 2 classes")
    n = y.size
    n_classes = classes.shape[0]
    class_weights = n / (n_classes * counts)
    weight_map = {cls: float(w) for cls, w in zip(classes, class_weights)}
    return np.array([weight_map[yi] for yi in y], dtype=float)


def roc_auc_score(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Return the area under the ROC curve for binary classification.

    The score is computed by sorting predictions by descending score and
    computing the Mann-Whitney U statistic, which equals the probability
    that a randomly chosen positive sample scores higher than a randomly
    chosen negative sample. Ties contribute 0.5.

    Parameters
    ----------
    y_true:
        Binary ground-truth labels (0 or 1).
    y_score:
        Continuous prediction scores (higher = more likely positive).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape")
    if y_true.size == 0:
        raise ValueError("at least one sample is required")
    labels = np.unique(y_true)
    if labels.shape[0] != 2:
        raise ValueError("roc_auc_score requires exactly 2 classes in y_true")
    pos = labels[1]
    neg = labels[0]
    pos_scores = y_score[y_true == pos]
    neg_scores = y_score[y_true == neg]
    if pos_scores.size == 0 or neg_scores.size == 0:
        raise ValueError("y_true must contain both positive and negative samples")
    diff = pos_scores[:, None] - neg_scores[None, :]
    n_pos = pos_scores.shape[0]
    n_neg = neg_scores.shape[0]
    auc = float(np.mean(np.sign(diff)) + 1.0) / 2.0
    return auc


def clone_estimator(estimator):
    """Return an unfitted shallow copy of an estimator.

    The clone reuses the ``__init__`` parameters read back from the instance,
    which works for every toolkit class that stores its constructor arguments
    as attributes with matching names.  Nested estimator parameters (e.g. a
    ``base_estimator``) are shared by reference rather than deep-copied.
    """
    cls = type(estimator)
    params = {}
    for name, param in inspect.signature(cls.__init__).parameters.items():
        if name == "self":
            continue
        if hasattr(estimator, name):
            params[name] = getattr(estimator, name)
        elif param.default is not inspect.Parameter.empty:
            params[name] = param.default
    return cls(**params)


def train_test_split(
    X: ArrayLike,
    y: ArrayLike,
    test_size: float = 0.25,
    random_state: Optional[int] = None,
    stratify: Optional[ArrayLike] = None,
) -> tuple:
    """Split arrays into random train and test subsets.

    When ``stratify`` is provided the class proportions are preserved in both
    splits, which keeps small datasets representative.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    indices = np.arange(n)

    if stratify is not None:
        stratify = np.asarray(stratify)
        train_idx: List[int] = []
        test_idx: List[int] = []
        for cls in np.unique(stratify):
            cls_idx = indices[stratify == cls]
            rng.shuffle(cls_idx)
            n_test = max(1, int(round(len(cls_idx) * test_size)))
            test_idx.extend(cls_idx[:n_test].tolist())
            train_idx.extend(cls_idx[n_test:].tolist())
        rng.shuffle(test_idx)
        rng.shuffle(train_idx)
    else:
        rng.shuffle(indices)
        n_test = int(round(n * test_size))
        n_test = min(max(n_test, 1), n - 1) if n > 1 else 0
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def accuracy_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def log_loss(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    labels: Optional[Sequence] = None,
) -> float:
    """Cross-entropy loss averaged over samples.

    ``y_prob`` must be a probability matrix whose columns line up with
    ``labels`` (or the sorted unique values of ``y_true`` when labels is
    omitted).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if labels is None:
        labels = np.unique(y_true)
    labels = np.asarray(labels)
    eps = 1e-15
    y_prob = np.clip(y_prob, eps, 1.0 - eps)
    n = y_true.shape[0]
    idx = np.searchsorted(labels, y_true)
    return float(-np.mean(np.log(y_prob[np.arange(n), idx])))


class _Node:
    """A single node in the decision tree."""

    __slots__ = ("feature", "threshold", "left", "right", "value", "is_leaf")

    def __init__(self) -> None:
        self.feature: Optional[int] = None
        self.threshold: Optional[float] = None
        self.left: Optional["_Node"] = None
        self.right: Optional["_Node"] = None
        self.value = None
        self.is_leaf = False


class DecisionTree:
    """A CART-like decision tree.

    Parameters
    ----------
    criterion :
        ``"gini"`` or ``"entropy"`` for classification, ``"variance"`` for
        regression.
    max_depth, min_samples_split, min_impurity_decrease :
        Stopping / pruning controls.
    max_features :
        Number of features to consider per split (int, ``"sqrt"`` or
        ``"log2"``).  ``None`` uses all features.
    random_state :
        Seed for reproducible feature sub-sampling.
    """

    def __init__(
        self,
        criterion: str = "gini",
        max_depth: int = 5,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.0,
        max_features: Optional[Union[int, str]] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.random_state = random_state
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: Optional[int] = None
        self.is_classifier_: bool = True
        self._tree: Optional[_Node] = None

    # ------------------------------------------------------------------ fit
    def fit(self, X: ArrayLike, y: ArrayLike) -> "DecisionTree":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y)
        self.is_classifier_ = self.criterion in ("gini", "entropy")
        rng = np.random.default_rng(self.random_state)
        if self.is_classifier_:
            self.classes_ = np.unique(y)
            self.n_classes_ = self.classes_.shape[0]
            class_to_int = {c: i for i, c in enumerate(self.classes_)}
            y_int = np.array([class_to_int[v] for v in y], dtype=np.intp)
            self._tree = self._build_classifier(X, y_int, depth=0, rng=rng)
        else:
            yf = np.asarray(y, dtype=np.float64)
            self._tree = self._build_regressor(X, yf, depth=0, rng=rng)
        return self

    # ------------------------------------------------------------- impurity
    def _impurity(self, y: np.ndarray) -> float:
        if self.is_classifier_:
            return self._gini(y) if self.criterion == "gini" else self._entropy(y)
        return self._variance(y)

    def _gini(self, y: np.ndarray) -> float:
        m = y.shape[0]
        if m == 0:
            return 0.0
        counts = np.bincount(y, minlength=self.n_classes_)
        p = counts / m
        return float(1.0 - np.sum(p ** 2))

    def _entropy(self, y: np.ndarray) -> float:
        m = y.shape[0]
        if m == 0:
            return 0.0
        counts = np.bincount(y, minlength=self.n_classes_)
        p = counts / m
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    def _variance(self, y: np.ndarray) -> float:
        if y.shape[0] == 0:
            return 0.0
        return float(np.var(y))

    # ----------------------------------------------------------- feature idx
    def _feature_indices(self, n_features: int, rng: np.random.Generator) -> np.ndarray:
        if self.max_features is None:
            return np.arange(n_features)
        if self.max_features == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            k = max(1, int(np.log2(n_features)))
        else:
            k = int(self.max_features)
            k = max(1, min(k, n_features))
        return rng.choice(n_features, size=k, replace=False)

    # ------------------------------------------------------- best split search
    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_indices: np.ndarray,
    ) -> Optional[dict]:
        """Find the best split using sorted cumulative statistics.

        Each candidate feature is sorted once (O(n log n)) and the gini/variance
        of every prefix is read from a running cumulative count, making the search
        O(n log n) per feature instead of O(n * thresholds).
        """
        n_samples = X.shape[0]
        parent_impurity = self._impurity(y)
        best_gain = self.min_impurity_decrease
        best: Optional[dict] = None

        for fi in feature_indices:
            x_f = X[:, fi]
            order = np.argsort(x_f, kind="mergesort")
            x_s = x_f[order]
            y_s = y[order]
            # split positions lie between adjacent distinct values
            distinct = np.nonzero(x_s[1:] != x_s[:-1])[0]
            if distinct.size == 0:
                continue
            if self.is_classifier_:
                onehot = np.zeros((n_samples, self.n_classes_))
                onehot[np.arange(n_samples), y_s] = 1.0
                cum = np.cumsum(onehot, axis=0)  # cum[i] = counts in prefix of len i+1
                total = cum[-1]
                for i in distinct:
                    t = i + 1
                    left = cum[i]
                    right = total - left
                    nL = left.sum()
                    nR = n_samples - nL
                    gini_l = 1.0 - np.sum((left / nL) ** 2)
                    gini_r = 1.0 - np.sum((right / nR) ** 2)
                    gain = parent_impurity - (nL / n_samples) * gini_l - (nR / n_samples) * gini_r
                    if gain > best_gain:
                        best_gain = gain
                        best = {
                            "feature": int(fi),
                            "threshold": float((x_s[i] + x_s[i + 1]) / 2.0),
                            "order": order,
                            "split_pos": t,
                        }
            else:
                csum = np.cumsum(y_s)
                csum2 = np.cumsum(y_s * y_s)
                total_sum = csum[-1]
                total_sum2 = csum2[-1]
                for i in distinct:
                    t = i + 1
                    sl = csum[i]
                    sl2 = csum2[i]
                    nL = t
                    nR = n_samples - t
                    sr = total_sum - sl
                    sr2 = total_sum2 - sl2
                    var_l = (sl2 - sl * sl / nL) / nL
                    var_r = (sr2 - sr * sr / nR) / nR
                    gain = parent_impurity - (nL / n_samples) * var_l - (nR / n_samples) * var_r
                    if gain > best_gain:
                        best_gain = gain
                        best = {
                            "feature": int(fi),
                            "threshold": float((x_s[i] + x_s[i + 1]) / 2.0),
                            "order": order,
                            "split_pos": t,
                        }
        return best

    # ------------------------------------------------------------- builders
    def _build_classifier(
        self, X: np.ndarray, y: np.ndarray, depth: int, rng: np.random.Generator
    ) -> _Node:
        node = _Node()
        n_samples = X.shape[0]
        counts = np.bincount(y, minlength=self.n_classes_ if self.n_classes_ else 0)
        node.value = counts
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
            or self._impurity(y) == 0.0
            or n_samples == 1
        ):
            node.is_leaf = True
            return node
        feature_indices = self._feature_indices(X.shape[1], rng)
        best = self._best_split(X, y, feature_indices)
        if best is None:
            node.is_leaf = True
            return node
        node.feature = best["feature"]
        node.threshold = best["threshold"]
        order = best["order"]
        t = best["split_pos"]
        node.left = self._build_classifier(X[order[:t]], y[order[:t]], depth + 1, rng)
        node.right = self._build_classifier(X[order[t:]], y[order[t:]], depth + 1, rng)
        return node

    def _build_regressor(
        self, X: np.ndarray, y: np.ndarray, depth: int, rng: np.random.Generator
    ) -> _Node:
        node = _Node()
        n_samples = X.shape[0]
        node.value = float(np.mean(y))
        if (
            (self.max_depth is not None and depth >= self.max_depth)
            or n_samples < self.min_samples_split
            or self._variance(y) == 0.0
            or n_samples == 1
        ):
            node.is_leaf = True
            return node
        feature_indices = self._feature_indices(X.shape[1], rng)
        best = self._best_split(X, y, feature_indices)
        if best is None:
            node.is_leaf = True
            return node
        node.feature = best["feature"]
        node.threshold = best["threshold"]
        order = best["order"]
        t = best["split_pos"]
        node.left = self._build_regressor(X[order[:t]], y[order[:t]], depth + 1, rng)
        node.right = self._build_regressor(X[order[t:]], y[order[t:]], depth + 1, rng)
        return node

    # --------------------------------------------------------------- predict
    def _traverse(self, row: np.ndarray) -> _Node:
        node = self._tree
        while not node.is_leaf:
            if row[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node

    def predict(self, X: ArrayLike) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.is_classifier_:
            proba = self.predict_proba(X)
            preds = proba.argmax(axis=1)
            return self.classes_[preds]
        return np.array([self._traverse(row).value for row in X])

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        if not self.is_classifier_:
            raise AttributeError("predict_proba is only defined for classification trees")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n = X.shape[0]
        proba = np.zeros((n, self.n_classes_))
        for i, row in enumerate(X):
            counts = self._traverse(row).value
            total = counts.sum()
            proba[i] = counts / total if total > 0 else 0.0
        return proba
