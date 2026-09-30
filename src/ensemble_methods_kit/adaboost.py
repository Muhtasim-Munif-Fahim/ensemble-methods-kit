"""AdaBoost: SAMME classifier and AdaBoost.R2 regressor."""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np

from .utils import DecisionTree

__all__ = ["AdaBoostClassifier", "AdaBoostRegressor"]


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


class AdaBoostRegressor:
    """AdaBoost.R2 ensemble of decision-tree weak regressors (Drucker, 1997).

    Each round fits a :class:`DecisionTree` with ``criterion="variance"``
    (a stump by default) on a bootstrap sample drawn according to the
    current sample weights. Absolute residuals are normalised by the
    maximum residual on that round; the average loss ``L`` yields
    ``beta = L / (1 - L)`` and a SAMME-style estimator weight
    ``alpha = learning_rate * log(1 / beta)``. Sample weights are then
    updated so hard examples receive more mass on the next round. The
    final prediction is the weighted median of the weak learners'
    outputs (weights ``log(1 / beta_t)``).

    Parameters
    ----------
    n_estimators :
        Maximum number of boosting rounds (= number of weak learners).
    learning_rate :
        Shrinkage applied to each estimator's weight (alpha).
    max_depth :
        Maximum depth of each weak learner. ``1`` yields decision stumps.
    loss :
        How absolute residuals are mapped to per-sample losses before
        averaging: ``"linear"`` (identity), ``"square"``, or
        ``"exponential"`` (``1 - exp(-e)``).
    random_state :
        Seed for reproducible bootstrap resampling.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        max_depth: int = 1,
        loss: str = "linear",
        random_state: Optional[int] = None,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if max_depth is not None and max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if loss not in ("linear", "square", "exponential"):
            raise ValueError('loss must be "linear", "square", or "exponential"')
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.loss = loss
        self.random_state = random_state
        self.estimators_: List[DecisionTree] = []
        self.estimator_weights_: List[float] = []
        self.estimator_errors_: List[float] = []
        self._n_features: Optional[int] = None

    def _loss_vector(self, errors: np.ndarray) -> np.ndarray:
        if self.loss == "linear":
            return errors
        if self.loss == "square":
            return errors ** 2
        # exponential
        return 1.0 - np.exp(-errors)

    def fit(self, X, y) -> "AdaBoostRegressor":
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y, dtype=float).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        n = X.shape[0]
        self._n_features = X.shape[1]
        self.estimators_ = []
        self.estimator_weights_ = []
        self.estimator_errors_ = []

        rng = np.random.default_rng(self.random_state)
        sample_weight = np.full(n, 1.0 / n)

        for _ in range(self.n_estimators):
            tree = DecisionTree(
                criterion="variance",
                max_depth=self.max_depth,
                random_state=self.random_state,
            )
            p = sample_weight / sample_weight.sum()
            indices = rng.choice(n, size=n, p=p)
            tree.fit(X[indices], y[indices])

            predictions = tree.predict(X).astype(float)
            abs_err = np.abs(predictions - y)
            max_err = float(np.max(abs_err))
            if max_err <= 1e-12:
                # Perfect fit on weighted sample — keep with large weight and stop.
                self.estimators_.append(tree)
                self.estimator_weights_.append(float(self.learning_rate * np.log(1.0 / 1e-16)))
                self.estimator_errors_.append(0.0)
                break

            errors = abs_err / max_err
            loss_vec = self._loss_vector(errors)
            # Average loss under current sample distribution.
            average_loss = float(np.dot(sample_weight, loss_vec) / sample_weight.sum())
            if average_loss >= 0.5:
                # Weak learner no better than random under AdaBoost.R2.
                if not self.estimators_:
                    # Keep at least one estimator so predict() works.
                    self.estimators_.append(tree)
                    self.estimator_weights_.append(1.0)
                    self.estimator_errors_.append(average_loss)
                break
            if average_loss <= 0.0:
                self.estimators_.append(tree)
                self.estimator_weights_.append(float(self.learning_rate * np.log(1.0 / 1e-16)))
                self.estimator_errors_.append(0.0)
                break

            average_loss = min(max(average_loss, 1e-16), 1.0 - 1e-16)
            beta = average_loss / (1.0 - average_loss)
            alpha = self.learning_rate * float(np.log(1.0 / beta))

            # w_i *= beta^(1 - e_i)  (Drucker / sklearn AdaBoost.R2)
            sample_weight *= np.power(beta, (1.0 - loss_vec) * self.learning_rate)
            sample_weight = np.clip(sample_weight, 1e-16, None)
            sample_weight /= sample_weight.sum()

            self.estimators_.append(tree)
            self.estimator_weights_.append(float(alpha))
            self.estimator_errors_.append(average_loss)

        return self

    def predict(self, X) -> np.ndarray:
        """Return the weighted-median prediction of the fitted estimators."""
        if not self.estimators_:
            raise RuntimeError("Estimator is not fitted yet")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # predictions: (n_samples, n_estimators)
        preds = np.column_stack([est.predict(X).astype(float) for est in self.estimators_])
        weights = np.asarray(self.estimator_weights_, dtype=float)
        # Stable weighted median per row.
        order = np.argsort(preds, axis=1)
        sorted_preds = np.take_along_axis(preds, order, axis=1)
        sorted_weights = weights[order]
        cum = np.cumsum(sorted_weights, axis=1)
        total = cum[:, -1:]
        # First index where cumulative weight reaches half the total mass.
        mask = cum >= (0.5 * total)
        # argmax of mask along axis=1 gives the first True
        median_idx = np.argmax(mask, axis=1)
        rows = np.arange(preds.shape[0])
        return sorted_preds[rows, median_idx]

    def staged_predict(self, X):
        """Yield weighted-median predictions after each boosting stage."""
        if not self.estimators_:
            raise RuntimeError("Estimator is not fitted yet")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        preds_all = [est.predict(X).astype(float) for est in self.estimators_]
        weights = np.asarray(self.estimator_weights_, dtype=float)
        for t in range(1, len(self.estimators_) + 1):
            preds = np.column_stack(preds_all[:t])
            w = weights[:t]
            order = np.argsort(preds, axis=1)
            sorted_preds = np.take_along_axis(preds, order, axis=1)
            sorted_weights = w[order]
            cum = np.cumsum(sorted_weights, axis=1)
            total = cum[:, -1:]
            mask = cum >= (0.5 * total)
            median_idx = np.argmax(mask, axis=1)
            rows = np.arange(preds.shape[0])
            yield sorted_preds[rows, median_idx]

    def _accumulate_split_features(self, node, weight: float, importances: np.ndarray) -> None:
        if node is None or node.is_leaf:
            return
        if node.feature is not None:
            importances[node.feature] += weight
        self._accumulate_split_features(node.left, weight, importances)
        self._accumulate_split_features(node.right, weight, importances)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Average weighted feature usage across all weak learners."""
        if not self.estimators_ or self._n_features is None:
            raise RuntimeError("Estimator is not fitted yet")
        importances = np.zeros(self._n_features)
        for est, alpha in zip(self.estimators_, self.estimator_weights_):
            self._accumulate_split_features(est._tree, alpha, importances)
        total = importances.sum()
        if total > 0:
            importances /= total
        return importances
