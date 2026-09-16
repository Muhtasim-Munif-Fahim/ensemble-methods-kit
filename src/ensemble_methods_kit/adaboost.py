"""AdaBoost (SAMME) classifier."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .utils import DecisionTree

__all__ = ["AdaBoostClassifier"]


class AdaBoostClassifier:
    """SAMME AdaBoost ensemble of decision-tree weak learners.

    Each round fits a :class:`DecisionTree` (a stump by default) on a
    bootstrap sample drawn according to the current sample weights.
    Misclassified samples receive higher weight so later rounds focus on
    hard cases.  The final prediction is a weighted majority vote — the
    discrete SAMME algorithm of Zhu et al., which reduces to classical
    AdaBoost.M1 when there are two classes.

    Parameters
    ----------
    n_estimators :
        Maximum number of boosting rounds (= number of weak learners).
    learning_rate :
        Shrinkage applied to each estimator's SAMME weight (alpha).
        Smaller values make the ensemble more conservative.
    max_depth :
        Maximum depth of each weak learner.  ``1`` yields decision stumps.
    random_state :
        Seed for reproducible bootstrap resampling.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        max_depth: int = 1,
        random_state: Optional[int] = None,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.estimators_: List[DecisionTree] = []
        self.weights_: List[float] = []
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: Optional[int] = None
        self._n_features: Optional[int] = None

    def fit(self, X, y) -> "AdaBoostClassifier":
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = int(self.classes_.shape[0])
        if n_classes < 2:
            raise ValueError("AdaBoost requires at least 2 classes")
        self.n_classes_ = n_classes

        n = X.shape[0]
        self._n_features = X.shape[1]
        self.estimators_ = []
        self.weights_ = []

        rng = np.random.default_rng(self.random_state)
        sample_weight = np.full(n, 1.0 / n)

        for _ in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth, random_state=self.random_state)
            p = sample_weight / sample_weight.sum()
            indices = rng.choice(n, size=n, p=p)
            tree.fit(X[indices], y[indices])

            predictions = tree.predict(X)
            incorrect = predictions != y
            err = float(np.dot(sample_weight, incorrect))

            # SAMME rejects a weak learner that is no better than random.
            if err >= 1.0 - 1.0 / n_classes:
                break
            if err <= 0.0:
                alpha = self.learning_rate * (
                    np.log((1.0 - 1e-16) / 1e-16) + np.log(n_classes - 1.0)
                )
                self.estimators_.append(tree)
                self.weights_.append(float(alpha))
                break

            err = min(max(err, 1e-16), 1.0 - 1e-16)
            alpha = self.learning_rate * (
                np.log((1.0 - err) / err) + np.log(n_classes - 1.0)
            )

            log_w = np.log(np.clip(sample_weight, 1e-15, None)) + alpha * incorrect
            log_w -= np.max(log_w)
            sample_weight = np.exp(log_w)
            sample_weight /= sample_weight.sum()

            self.estimators_.append(tree)
            self.weights_.append(float(alpha))

        return self

    def _class_scores(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        scores = np.zeros((n, self.n_classes_))
        for est, alpha in zip(self.estimators_, self.weights_):
            pred = est.predict(X)
            for k, label in enumerate(self.classes_):
                scores[pred == label, k] += alpha
        return scores

    def predict(self, X) -> np.ndarray:
        if not self.estimators_:
            raise RuntimeError("Estimator is not fitted yet")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        scores = self._class_scores(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X) -> np.ndarray:
        if not self.estimators_:
            raise RuntimeError("Estimator is not fitted yet")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        scores = self._class_scores(X)
        total = float(sum(self.weights_))
        n_classes = self.n_classes_
        if total <= 0.0 or n_classes is None or n_classes < 2:
            return np.full((X.shape[0], n_classes or 1), 1.0 / (n_classes or 1))
        # Zhu et al. SAMME: softmax of the normalised weighted votes.
        proba = np.exp((1.0 / (n_classes - 1.0)) * (scores / total))
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    @property
    def estimator_weights_(self) -> List[float]:
        return self.weights_

    def _accumulate_split_features(self, node, weight: float, importances: np.ndarray) -> None:
        if node is None or node.is_leaf:
            return
        if node.feature is not None:
            importances[node.feature] += weight
        self._accumulate_split_features(node.left, weight, importances)
        self._accumulate_split_features(node.right, weight, importances)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Average weighted feature usage across all weak learners.

        Each split feature accumulates the parent estimator's SAMME weight
        (alpha).  Features never selected by any learner receive 0.0.
        The returned array is normalised to sum to 1.
        """
        if not self.estimators_ or self._n_features is None:
            raise RuntimeError("Estimator is not fitted yet")
        importances = np.zeros(self._n_features)
        for est, alpha in zip(self.estimators_, self.weights_):
            self._accumulate_split_features(est._tree, alpha, importances)
        total = importances.sum()
        if total > 0:
            importances /= total
        return importances
