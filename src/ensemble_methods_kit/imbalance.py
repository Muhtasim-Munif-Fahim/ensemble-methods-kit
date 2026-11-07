"""Imbalanced-learning ensembles: EasyEnsemble, BalanceCascade, and RUSBoost.

EasyEnsemble (Liu, Wu & Zhou, 2009) trains ``n_subsets`` independent
AdaBoost-style (or bagged-tree) classifiers, each on a balanced bootstrap
that keeps *all* minority-class samples and undersamples the majority class
to the same size.  Predictions are soft-averaged across subsets.

BalanceCascade iteratively removes majority samples that are confidently
classified by the current subset ensemble, so later subsets focus on harder
majority examples.

RUSBoost (Seiffert, Khoshgoftaar, Van Hulse & Napolitano, 2010) wraps
SAMME AdaBoost around random undersampling: each boosting round draws a
balanced temporary training set (all minority samples plus a random majority
subset of the same size), fits a weak tree on it, then updates AdaBoost
weights on the full training set so hard examples of either class keep
rising in importance.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .utils import DecisionTree

__all__ = ["EasyEnsembleClassifier", "BalanceCascadeClassifier", "RUSBoostClassifier"]


def _majority_minority(y: np.ndarray):
    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 2:
        raise ValueError("need at least two classes for EasyEnsemble / BalanceCascade / RUSBoost")
    order = np.argsort(counts)
    minority = classes[order[0]]
    majority = classes[order[-1]]
    return majority, minority, classes


class EasyEnsembleClassifier:
    """EasyEnsemble undersampling bagging for binary / multi-class imbalance.

    Parameters
    ----------
    n_subsets :
        Number of balanced subsets (and therefore base ensembles).
    n_estimators :
        Trees trained inside each subset (bagged, soft-voted).
    max_depth :
        Max depth of each decision tree.
    random_state :
        Seed for reproducible undersampling.
    """

    def __init__(
        self,
        n_subsets: int = 10,
        n_estimators: int = 10,
        max_depth: Optional[int] = 5,
        random_state: Optional[int] = None,
    ) -> None:
        if n_subsets < 1:
            raise ValueError("n_subsets must be >= 1")
        if n_estimators < 1:
            raise ValueError("n_estimators must be >= 1")
        self.n_subsets = int(n_subsets)
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.random_state = random_state
        self.estimators_: list = []
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: Optional[int] = None

    def fit(self, X, y) -> "EasyEnsembleClassifier":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = int(self.classes_.shape[0])
        majority, minority, _ = _majority_minority(y)
        min_idx = np.flatnonzero(y == minority)
        maj_idx = np.flatnonzero(y == majority)
        n_min = int(min_idx.size)
        if n_min == 0 or maj_idx.size == 0:
            raise ValueError("both majority and minority classes must be non-empty")
        # For multi-class: keep minority fully; undersample every other class
        # to n_min as well (one-vs-rest style balance per subset).
        other = {c: np.flatnonzero(y == c) for c in self.classes_ if c != minority}
        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        for _ in range(self.n_subsets):
            parts = [min_idx]
            for c, idx in other.items():
                take = min(n_min, int(idx.size))
                chosen = rng.choice(idx, size=take, replace=False)
                parts.append(chosen)
            sample_idx = np.concatenate(parts)
            rng.shuffle(sample_idx)
            Xs, ys = X[sample_idx], y[sample_idx]
            trees = []
            for _t in range(self.n_estimators):
                boot = rng.integers(0, Xs.shape[0], size=Xs.shape[0])
                tree = DecisionTree(
                    criterion="gini",
                    max_depth=self.max_depth,
                    min_samples_split=2,
                    max_features="sqrt",
                    random_state=int(rng.integers(0, 2**31 - 1)),
                )
                tree.fit(Xs[boot], ys[boot])
                trees.append(tree)
            self.estimators_.append(trees)
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if not self.estimators_:
            raise RuntimeError("EasyEnsembleClassifier must be fitted before predict")
        subset_probas = []
        for trees in self.estimators_:
            proba = np.mean([t.predict_proba(X) for t in trees], axis=0)
            subset_probas.append(proba)
        return np.mean(subset_probas, axis=0)

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class BalanceCascadeClassifier:
    """BalanceCascade: iterative EasyEnsemble that drops easy majority samples.

    After each subset is trained, majority samples whose predicted
    probability for the majority class exceeds ``removal_threshold`` are
    removed from the pool for subsequent subsets.
    """

    def __init__(
        self,
        n_subsets: int = 10,
        n_estimators: int = 10,
        max_depth: Optional[int] = 5,
        removal_threshold: float = 0.5,
        random_state: Optional[int] = None,
    ) -> None:
        if n_subsets < 1:
            raise ValueError("n_subsets must be >= 1")
        if not 0.0 < removal_threshold < 1.0:
            raise ValueError("removal_threshold must be in (0, 1)")
        self.n_subsets = int(n_subsets)
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.removal_threshold = float(removal_threshold)
        self.random_state = random_state
        self.estimators_: list = []
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: Optional[int] = None

    def fit(self, X, y) -> "BalanceCascadeClassifier":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = int(self.classes_.shape[0])
        majority, minority, _ = _majority_minority(y)
        maj_class_pos = int(np.where(self.classes_ == majority)[0][0])
        min_idx = np.flatnonzero(y == minority)
        maj_pool = np.flatnonzero(y == majority).tolist()
        n_min = int(min_idx.size)
        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        for _ in range(self.n_subsets):
            if not maj_pool:
                break
            take = min(n_min, len(maj_pool))
            chosen_maj = rng.choice(np.asarray(maj_pool), size=take, replace=False)
            sample_idx = np.concatenate([min_idx, chosen_maj])
            rng.shuffle(sample_idx)
            Xs, ys = X[sample_idx], y[sample_idx]
            trees = []
            for _t in range(self.n_estimators):
                boot = rng.integers(0, Xs.shape[0], size=Xs.shape[0])
                tree = DecisionTree(
                    criterion="gini",
                    max_depth=self.max_depth,
                    min_samples_split=2,
                    max_features="sqrt",
                    random_state=int(rng.integers(0, 2**31 - 1)),
                )
                tree.fit(Xs[boot], ys[boot])
                trees.append(tree)
            self.estimators_.append(trees)
            # Drop easy majority samples from the remaining pool
            if not maj_pool:
                break
            maj_X = X[np.asarray(maj_pool)]
            proba = np.mean([t.predict_proba(maj_X) for t in trees], axis=0)
            keep = []
            for i, idx in enumerate(maj_pool):
                if proba[i, maj_class_pos] < self.removal_threshold:
                    keep.append(idx)
            maj_pool = keep
        if not self.estimators_:
            raise RuntimeError("BalanceCascade produced no subsets")
        return self

    def predict_proba(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if not self.estimators_:
            raise RuntimeError("BalanceCascadeClassifier must be fitted before predict")
        subset_probas = []
        for trees in self.estimators_:
            proba = np.mean([t.predict_proba(X) for t in trees], axis=0)
            subset_probas.append(proba)
        return np.mean(subset_probas, axis=0)

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class RUSBoostClassifier:
    """RUSBoost: Random UnderSampling inside SAMME AdaBoost (Seiffert et al. 2010).

    Each boosting round undersamples every non-minority class down to the
    minority count (``sampling_strategy="auto"``) or, when a float in
    ``(0, 1]`` is given, keeps ``ceil(strategy * n_class)`` samples of each
    non-minority class (``1.0`` matches the minority size). A weak
    decision tree is fit on that temporary balanced set; AdaBoost sample
    weights are then updated on the *full* training set from the SAMME
    weighted error.

    Parameters
    ----------
    n_estimators :
        Maximum boosting rounds.
    learning_rate :
        Shrinkage on each weak learner SAMME weight.
    max_depth :
        Depth of each weak tree (``1`` = stump).
    sampling_strategy :
        ``"auto"`` or a float in ``(0, 1]`` (see above).
    random_state :
        Seed for undersampling and bootstrap draws.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        max_depth: int = 1,
        sampling_strategy: float | str = "auto",
        random_state: Optional[int] = None,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if isinstance(sampling_strategy, str):
            if sampling_strategy != "auto":
                raise ValueError('sampling_strategy must be "auto" or a float in (0, 1]')
        elif isinstance(sampling_strategy, bool) or not isinstance(sampling_strategy, (int, float)):
            raise ValueError('sampling_strategy must be "auto" or a float in (0, 1]')
        elif not 0.0 < float(sampling_strategy) <= 1.0:
            raise ValueError('sampling_strategy must be "auto" or a float in (0, 1]')
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = max_depth
        self.sampling_strategy = (
            sampling_strategy if isinstance(sampling_strategy, str) else float(sampling_strategy)
        )
        self.random_state = random_state
        self.estimators_: list = []
        self.estimator_weights_: list = []
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: Optional[int] = None

    def _undersample_indices(self, y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        majority, minority, classes = _majority_minority(y)
        min_idx = np.flatnonzero(y == minority)
        n_min = int(min_idx.size)
        parts = [min_idx]
        for c in classes:
            if c == minority:
                continue
            idx = np.flatnonzero(y == c)
            if self.sampling_strategy == "auto" or (
                isinstance(self.sampling_strategy, float)
                and abs(self.sampling_strategy - 1.0) < 1e-12
            ):
                take = min(n_min, int(idx.size))
            else:
                take = max(1, min(int(idx.size), int(np.ceil(float(self.sampling_strategy) * int(idx.size)))))
            chosen = rng.choice(idx, size=take, replace=False)
            parts.append(chosen)
        sample_idx = np.concatenate(parts)
        rng.shuffle(sample_idx)
        return sample_idx

    def fit(self, X, y) -> "RUSBoostClassifier":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = int(self.classes_.shape[0])
        if n_classes < 2:
            raise ValueError(
                "need at least two classes for EasyEnsemble / BalanceCascade / RUSBoost"
            )
        self.n_classes_ = n_classes
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        sample_weight = np.full(n, 1.0 / n)
        self.estimators_ = []
        self.estimator_weights_ = []

        for _ in range(self.n_estimators):
            rus_idx = self._undersample_indices(y, rng)
            w_sub = sample_weight[rus_idx]
            w_sub = w_sub / w_sub.sum()
            boot = rng.choice(rus_idx, size=len(rus_idx), replace=True, p=w_sub)
            tree = DecisionTree(
                criterion="gini",
                max_depth=self.max_depth,
                min_samples_split=2,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            tree.fit(X[boot], y[boot])

            predictions = tree.predict(X)
            incorrect = predictions != y
            err = float(np.dot(sample_weight, incorrect))
            if err >= 1.0 - 1.0 / n_classes:
                break
            if err <= 0.0:
                alpha = self.learning_rate * (
                    np.log((1.0 - 1e-16) / 1e-16) + np.log(n_classes - 1.0)
                )
                self.estimators_.append(tree)
                self.estimator_weights_.append(float(alpha))
                break
            err = min(max(err, 1e-16), 1.0 - 1e-16)
            alpha = self.learning_rate * (
                np.log((1.0 - err) / err) + np.log(n_classes - 1.0)
            )
            sample_weight = sample_weight * np.exp(alpha * incorrect.astype(float))
            sample_weight = np.clip(sample_weight, 1e-16, None)
            sample_weight /= sample_weight.sum()
            self.estimators_.append(tree)
            self.estimator_weights_.append(float(alpha))

        if not self.estimators_:
            raise RuntimeError("RUSBoost produced no estimators")
        return self

    def decision_function(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if not self.estimators_:
            raise RuntimeError("RUSBoostClassifier must be fitted before predict")
        n = X.shape[0]
        scores = np.zeros((n, self.n_classes_))
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        for tree, alpha in zip(self.estimators_, self.estimator_weights_):
            pred = tree.predict(X)
            for i, label in enumerate(pred):
                scores[i, class_to_idx[label]] += alpha
        return scores

    def predict_proba(self, X) -> np.ndarray:
        scores = self.decision_function(X)
        scores = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]
