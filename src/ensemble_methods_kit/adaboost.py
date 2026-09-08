"""AdaBoost (Adaptive Boosting) classifier."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .utils import DecisionTree

__all__ = ["AdaBoostClassifier"]


class AdaBoostClassifier:
    """Adaptive Boosting ensemble of shallow decision stumps.

    Each round fits a weak classifier (a :class:`DecisionTree` with
    ``max_depth=1``) on a bootstrap sample drawn according to the current
    sample weights. Samples misclassified by earlier rounds receive higher
    weights, forcing later rounds to focus on hard cases. The final
    prediction is a weighted majority vote.

    Parameters
    ----------
    n_estimators :
        Maximum number of boosting rounds (= number of stumps).
    learning_rate :
        Shrinkage applied to each stump's weight (alpha). Smaller values
        make the ensemble more conservative.
    random_state :
        Seed for reproducible bootstrap resampling.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        random_state: Optional[int] = None,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.estimators_: List[DecisionTree] = []
        self.weights_: List[float] = []
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X, y) -> "AdaBoostClassifier":
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("AdaBoost supports binary classification only")

        n = X.shape[0]
        pos_label = self.classes_[1]
        y_binary = (y == pos_label).astype(int)

        rng = np.random.default_rng(self.random_state)
        weights = np.full(n, 1.0 / n)

        for _ in range(self.n_estimators):
            stump = DecisionTree(max_depth=1, random_state=self.random_state)
            indices = rng.choice(n, size=n, p=weights)
            X_boot = X[indices]
            y_boot = y_binary[indices]
            stump.fit(X_boot, y_boot)

            predictions = stump.predict(X).astype(int)
            incorrect = (predictions != y_binary).astype(float)
            err = np.dot(weights, incorrect)

            if err >= 0.5:
                break
            if err == 0:
                alpha = 10.0
            else:
                alpha = self.learning_rate * 0.5 * np.log((1.0 - err) / err)

            weights *= np.exp(-alpha * (2 * y_binary - 1) * (2 * predictions - 1))
            weights /= np.sum(weights)

            self.estimators_.append(stump)
            self.weights_.append(alpha)

        return self

    def predict(self, X) -> np.ndarray:
        if not self.estimators_:
            raise RuntimeError("Estimator is not fitted yet")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        pos_label = self.classes_[1]

        votes = np.zeros(X.shape[0])
        for est, alpha in zip(self.estimators_, self.weights_):
            pred = est.predict(X).astype(int)
            votes += alpha * (2 * pred - 1)

        return np.where(votes >= 0, pos_label, self.classes_[0])

    def predict_proba(self, X) -> np.ndarray:
        if not self.estimators_:
            raise RuntimeError("Estimator is not fitted yet")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n = X.shape[0]
        total_weight = sum(self.weights_)
        proba_positive = np.zeros(n)

        for est, alpha in zip(self.estimators_, self.weights_):
            pred = est.predict(X).astype(int)
            proba_positive += alpha * pred

        proba_positive /= total_weight
        return np.column_stack([1.0 - proba_positive, proba_positive])

    @property
    def estimator_weights_(self) -> List[float]:
        return self.weights_
