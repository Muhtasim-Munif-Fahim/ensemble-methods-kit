"""Caruana ensemble selection: greedy forward selection from a model library.

Caruana, Niculescu-Mizil, Crew & Ksikes (2004), "Ensemble Selection from
Libraries of Models".  A library of heterogeneous base models is fitted on a
training split.  Starting from the best ``n_init`` models, the ensemble is grown
one model at a time by adding (with replacement) whichever library member most
improves the hill-climbing metric of the averaged validation predictions.
Selection counts become the ensemble weights, so a model picked three times
gets three times the weight of a model picked once.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .utils import clone_estimator, train_test_split

__all__ = ["EnsembleSelectionClassifier", "EnsembleSelectionRegressor"]

_EPS = 1e-15


def _as_2d(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
    return X


def _validate_library(estimators, need_proba: bool) -> List[Tuple[str, object]]:
    estimators = list(estimators)
    if len(estimators) == 0 or not all(
        isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
        for item in estimators
    ):
        raise ValueError("estimators must be a non-empty list of (name, estimator) pairs")
    names = [name for name, _ in estimators]
    if len(names) != len(set(names)):
        raise ValueError("estimator names must be unique")
    for name, est in estimators:
        if not hasattr(est, "fit") or not hasattr(est, "predict"):
            raise ValueError(f"estimator {name!r} must implement fit and predict")
        if need_proba and not hasattr(est, "predict_proba"):
            raise ValueError(f"estimator {name!r} must implement predict_proba")
    return estimators


class _EnsembleSelectionBase:
    """Shared greedy hill-climbing logic for classifier and regressor."""

    _metrics: Tuple[str, ...] = ()

    def __init__(
        self,
        estimators: Sequence[Tuple[str, object]],
        n_iterations: int = 50,
        n_init: int = 1,
        validation_fraction: float = 0.25,
        metric: Optional[str] = None,
        use_best: bool = True,
        refit: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        self.estimators = list(estimators)
        self.n_iterations = n_iterations
        self.n_init = n_init
        self.validation_fraction = validation_fraction
        self.metric = metric
        self.use_best = use_best
        self.refit = refit
        self.random_state = random_state
        self.estimators_: Optional[List] = None
        self.weights_: Optional[np.ndarray] = None
        self.selection_counts_: Optional[np.ndarray] = None
        self.selection_order_: Optional[List[int]] = None
        self.validation_scores_: Optional[np.ndarray] = None
        self.best_score_: Optional[float] = None
        self.n_features_in_: Optional[int] = None
        self._check_params()

    # -- validation -------------------------------------------------------
    def _check_params(self) -> None:
        _validate_library(self.estimators, self._need_proba())
        if not isinstance(self.n_iterations, (int, np.integer)) or self.n_iterations < 0:
            raise ValueError("n_iterations must be a non-negative integer")
        if not isinstance(self.n_init, (int, np.integer)) or self.n_init < 1:
            raise ValueError("n_init must be a positive integer")
        if self.n_init > len(self.estimators):
            raise ValueError("n_init cannot exceed the number of estimators")
        if not 0.0 < float(self.validation_fraction) < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")
        if self._metric_name() not in self._metrics:
            raise ValueError(f"metric must be one of {self._metrics}")

    def _metric_name(self) -> str:
        return self._metrics[0] if self.metric is None else self.metric

    def _need_proba(self) -> bool:  # pragma: no cover - overridden
        raise NotImplementedError

    def _check_fitted(self) -> None:
        if self.estimators_ is None or self.weights_ is None:
            raise RuntimeError("Estimator is not fitted yet")

    def _validate_X(self, X) -> np.ndarray:
        X = _as_2d(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {type(self).__name__} is expecting "
                f"{self.n_features_in_} features as input"
            )
        return X

    # -- core -------------------------------------------------------------
    def _greedy_select(self, preds: np.ndarray, y_val: np.ndarray) -> None:
        """Run forward selection with replacement over library predictions.

        ``preds`` has shape ``(n_models, n_val, ...)``.  Lower losses win.
        """
        n_models = preds.shape[0]
        single = np.array([self._loss(y_val, preds[m]) for m in range(n_models)])
        order = list(np.argsort(single, kind="stable")[: self.n_init])
        running = preds[order].sum(axis=0)
        trace = [self._loss(y_val, running / len(order))]
        for _ in range(int(self.n_iterations)):
            size = len(order) + 1
            losses = np.array(
                [self._loss(y_val, (running + preds[m]) / size) for m in range(n_models)]
            )
            best = int(np.argmin(losses))
            order.append(best)
            running = running + preds[best]
            trace.append(float(losses[best]))
        trace_arr = np.asarray(trace, dtype=np.float64)
        stop = len(order)
        if self.use_best:
            # trace[i] is the loss of the ensemble holding order[: n_init + i]
            stop = self.n_init + int(np.argmin(trace_arr))
        chosen = order[:stop]
        counts = np.bincount(np.asarray(chosen, dtype=np.intp), minlength=n_models)
        self.selection_order_ = [int(i) for i in chosen]
        self.selection_counts_ = counts
        self.weights_ = counts / counts.sum()
        self.validation_scores_ = trace_arr
        self.best_score_ = float(trace_arr[stop - self.n_init])

    def _fit_library(self, X, y) -> List:
        fitted = []
        for _, est in self.estimators:
            clone = clone_estimator(est)
            clone.fit(X, y)
            fitted.append(clone)
        return fitted

    @property
    def named_weights_(self) -> dict:
        """Mapping from estimator name to its ensemble weight."""
        self._check_fitted()
        return {name: float(w) for (name, _), w in zip(self.estimators, self.weights_)}


class EnsembleSelectionClassifier(_EnsembleSelectionBase):
    """Caruana-style greedy ensemble selection over a library of classifiers.

    Parameters
    ----------
    estimators :
        Non-empty list of ``(name, estimator)`` pairs. Each estimator must
        implement ``fit``, ``predict`` and ``predict_proba``. They are cloned,
        so the passed objects stay unfitted.
    n_iterations :
        Number of greedy additions after the initial ``n_init`` models.
    n_init :
        Sorted initialisation: the ensemble starts with the ``n_init`` best
        single models on the validation split.
    validation_fraction :
        Fraction of rows held out (stratified) for hill climbing.
    metric :
        ``"log_loss"`` (default), ``"brier"``, or ``"error"`` (1 - accuracy).
    use_best :
        Truncate the selection path at its best validation score (Caruana's
        recommendation) instead of keeping all ``n_iterations`` additions.
    refit :
        Refit every library member on the full data after the weights have
        been chosen on the validation split.
    random_state :
        Seed for the train/validation split.

    Attributes
    ----------
    weights_ : ndarray of shape (n_estimators,)
        Normalised selection counts; zero for models never selected.
    selection_order_ : list of int
        Library indices in the order they were added.
    validation_scores_ : ndarray
        Validation loss after the initialisation and after each addition.
    """

    _metrics = ("log_loss", "brier", "error")

    def _need_proba(self) -> bool:
        return True

    def _align(self, estimator, proba: np.ndarray) -> np.ndarray:
        proba = np.asarray(proba, dtype=np.float64)
        est_classes = getattr(estimator, "classes_", None)
        n_classes = self.classes_.shape[0]
        if est_classes is None:
            if proba.shape[1] != n_classes:
                raise ValueError("predict_proba column count does not match classes_")
            return proba
        est_classes = np.asarray(est_classes)
        if est_classes.shape[0] == n_classes and np.array_equal(est_classes, self.classes_):
            return proba
        aligned = np.zeros((proba.shape[0], n_classes), dtype=np.float64)
        index = {label: i for i, label in enumerate(self.classes_.tolist())}
        for column, label in enumerate(est_classes.tolist()):
            if label not in index:
                raise ValueError(f"estimator class {label!r} was not in the training labels")
            aligned[:, index[label]] = proba[:, column]
        return aligned

    def _loss(self, y_idx: np.ndarray, proba: np.ndarray) -> float:
        metric = self._metric_name()
        n = y_idx.shape[0]
        if metric == "log_loss":
            p = np.clip(proba[np.arange(n), y_idx], _EPS, 1.0)
            return float(-np.mean(np.log(p)))
        if metric == "brier":
            onehot = np.zeros_like(proba)
            onehot[np.arange(n), y_idx] = 1.0
            return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))
        return float(np.mean(np.argmax(proba, axis=1) != y_idx))

    def fit(self, X, y) -> "EnsembleSelectionClassifier":
        """Fit the library, hill-climb the weights, and optionally refit."""
        self._check_params()
        X = _as_2d(X)
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y must be 1D with one label per row of X")
        self.classes_ = np.unique(y)
        if self.classes_.shape[0] < 2:
            raise ValueError("EnsembleSelectionClassifier requires at least 2 classes")
        self.n_features_in_ = int(X.shape[1])
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=float(self.validation_fraction),
            random_state=self.random_state, stratify=y,
        )
        if X_tr.shape[0] == 0:
            raise ValueError("training split is empty; lower validation_fraction")
        library = self._fit_library(X_tr, y_tr)
        preds = np.stack(
            [self._align(est, est.predict_proba(X_val)) for est in library], axis=0
        )
        y_idx = np.searchsorted(self.classes_, y_val)
        self._greedy_select(preds, y_idx)
        self.estimators_ = self._fit_library(X, y) if self.refit else library
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Weighted average of the selected models' class probabilities."""
        self._check_fitted()
        X = self._validate_X(X)
        out = np.zeros((X.shape[0], self.classes_.shape[0]), dtype=np.float64)
        for weight, est in zip(self.weights_, self.estimators_):
            if weight > 0:
                out += weight * self._align(est, est.predict_proba(X))
        return out

    def predict(self, X) -> np.ndarray:
        """Predict the class with the highest ensemble probability."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


class EnsembleSelectionRegressor(_EnsembleSelectionBase):
    """Caruana-style greedy ensemble selection over a library of regressors.

    Same parameters as :class:`EnsembleSelectionClassifier`, except ``metric``
    is ``"mse"`` (default) or ``"mae"`` and the validation split is random
    rather than stratified.
    """

    _metrics = ("mse", "mae")

    def _need_proba(self) -> bool:
        return False

    def _loss(self, y_val: np.ndarray, pred: np.ndarray) -> float:
        resid = y_val - pred
        if self._metric_name() == "mse":
            return float(np.mean(resid ** 2))
        return float(np.mean(np.abs(resid)))

    def fit(self, X, y) -> "EnsembleSelectionRegressor":
        """Fit the library, hill-climb the weights, and optionally refit."""
        self._check_params()
        X = _as_2d(X)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if X.shape[0] < 2:
            raise ValueError("at least two samples are required")
        self.n_features_in_ = int(X.shape[1])
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=float(self.validation_fraction), random_state=self.random_state
        )
        library = self._fit_library(X_tr, y_tr)
        preds = np.stack(
            [np.asarray(est.predict(X_val), dtype=np.float64).reshape(-1) for est in library],
            axis=0,
        )
        self._greedy_select(preds, y_val)
        self.estimators_ = self._fit_library(X, y) if self.refit else library
        return self

    def predict(self, X) -> np.ndarray:
        """Weighted average of the selected models' predictions."""
        self._check_fitted()
        X = self._validate_X(X)
        out = np.zeros(X.shape[0], dtype=np.float64)
        for weight, est in zip(self.weights_, self.estimators_):
            if weight > 0:
                out += weight * np.asarray(est.predict(X), dtype=np.float64).reshape(-1)
        return out

    def score(self, X, y) -> float:
        """Coefficient of determination R² on ``(X, y)``."""
        from .utils import r2_score

        return r2_score(y, self.predict(X))
