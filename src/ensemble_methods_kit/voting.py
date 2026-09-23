"""Voting ensembles that combine independent classifiers."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .utils import clone_estimator

__all__ = ["VotingClassifier"]


def _as_2d(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
    return X


def _fit_estimator(estimator, X, y, sample_weight):
    """Fit ``estimator``, forwarding ``sample_weight`` when it is provided."""
    if sample_weight is None:
        estimator.fit(X, y)
        return estimator
    try:
        estimator.fit(X, y, sample_weight=sample_weight)
    except TypeError as exc:
        if "sample_weight" in str(exc):
            raise TypeError(
                f"Underlying estimator {type(estimator).__name__} does not "
                "support sample weights."
            ) from exc
        raise
    return estimator


class VotingClassifier:
    """Combine classifiers by majority vote or by averaging probabilities.

    ``fit`` clones every base estimator, so the objects passed in
    ``estimators`` are left unfitted. Soft voting lines each model's
    ``predict_proba`` columns up with :attr:`classes_` before averaging, which
    keeps the vote correct when a base learner reports classes in a different
    order.

    Parameters
    ----------
    estimators :
        Non-empty list of ``(name, estimator)`` pairs. A pair whose estimator
        is the string ``"drop"`` is skipped. Names must be unique strings.
        Each kept estimator must implement ``fit`` and ``predict``. Soft
        voting also requires ``predict_proba``.
    voting :
        ``"soft"`` (default) averages class probabilities. ``"hard"`` takes a
        weighted majority of the predicted labels. Ties resolve to the
        earliest label in ``classes_`` (sorted unique training labels).
    weights :
        One weight per entry in ``estimators``, including dropped entries.
        Dropped estimators do not contribute. ``None`` uses equal weights.
    n_jobs :
        Accepted for a scikit-learn-style signature. Fitting is serial.
    flatten_transform :
        When ``voting="soft"``, ``True`` makes :meth:`transform` return
        ``(n_samples, n_estimators * n_classes)``. ``False`` returns
        ``(n_estimators, n_samples, n_classes)``. Ignored for hard voting.
    """

    def __init__(
        self,
        estimators: Sequence[Tuple[str, object]],
        voting: str = "soft",
        weights: Optional[Sequence[float]] = None,
        n_jobs: Optional[int] = None,
        flatten_transform: bool = True,
    ) -> None:
        if voting not in ("hard", "soft"):
            raise ValueError("voting must be 'hard' or 'soft'")
        if not isinstance(flatten_transform, bool):
            raise ValueError("flatten_transform must be a bool")
        self.estimators = list(estimators)
        self.voting = voting
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float64)
        self.n_jobs = n_jobs
        self.flatten_transform = flatten_transform
        self.named_estimators_: dict = {}
        self.estimators_: Optional[List] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: Optional[int] = None
        self.n_features_in_: Optional[int] = None
        self._validate_estimators()
        self._weights_vec()

    def _validate_estimators(self) -> List[Tuple[str, object]]:
        if len(self.estimators) == 0 or not all(
            isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
            for item in self.estimators
        ):
            raise ValueError(
                "estimators must be a non-empty list of (name, estimator) pairs"
            )
        names = [name for name, _ in self.estimators]
        if len(names) != len(set(names)):
            raise ValueError("estimator names must be unique")
        active = [(name, est) for name, est in self.estimators if est != "drop"]
        if not active:
            raise ValueError("at least one estimator is required")
        for name, est in active:
            if not hasattr(est, "fit") or not hasattr(est, "predict"):
                raise ValueError(
                    f"estimator {name!r} must implement fit and predict"
                )
            if self.voting == "soft" and not hasattr(est, "predict_proba"):
                raise ValueError(
                    f"soft voting requires predict_proba; estimator {name!r} "
                    "does not define it"
                )
        return active

    def _weights_vec(self) -> np.ndarray:
        active_mask = [est != "drop" for _, est in self.estimators]
        n_active = sum(active_mask)
        if self.weights is None:
            return np.ones(n_active, dtype=np.float64)
        weights = np.asarray(self.weights, dtype=np.float64)
        if weights.ndim != 1 or weights.shape[0] != len(self.estimators):
            raise ValueError(
                "weights must contain one finite value per estimator "
                f"(got {weights.size}, expected {len(self.estimators)})"
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights must be finite")
        active = weights[np.asarray(active_mask)]
        if np.isclose(active.sum(), 0.0):
            raise ValueError("weights of kept estimators must not sum to zero")
        return active

    def _check_fitted(self) -> None:
        if self.estimators_ is None or self.classes_ is None:
            raise RuntimeError("Estimator is not fitted yet")

    def _validate_X(self, X) -> np.ndarray:
        X = _as_2d(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but VotingClassifier is expecting "
                f"{self.n_features_in_} features as input"
            )
        return X

    def fit(self, X, y, sample_weight=None) -> "VotingClassifier":
        """Fit a clone of each kept base estimator.

        Parameters
        ----------
        X, y :
            Training features and discrete class labels.
        sample_weight :
            Per-sample weights forwarded to each base ``fit``. Estimators that
            do not accept ``sample_weight`` raise ``TypeError``.
        """
        active = self._validate_estimators()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array of shape (n_samples, n_features)")
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.ravel()
        if y.ndim != 1:
            raise ValueError("y must be a 1D array of class labels")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.shape != (X.shape[0],):
                raise ValueError("sample_weight must have one value per sample")

        classes = np.unique(y)
        n_classes = int(classes.shape[0])
        if n_classes < 2:
            raise ValueError("VotingClassifier requires at least 2 classes")
        # Touch weights during fit so a bad vector fails before any training.
        self._weights_vec()

        fitted: List = []
        named: dict = {}
        for name, est in self.estimators:
            if est == "drop":
                named[name] = "drop"
                continue
            cloned = clone_estimator(est)
            _fit_estimator(cloned, X, y, sample_weight)
            fitted.append(cloned)
            named[name] = cloned
        if len(fitted) != len(active):
            raise RuntimeError("failed to fit every kept estimator")
        self.classes_ = classes
        self.n_classes_ = n_classes
        self.n_features_in_ = int(X.shape[1])
        self.estimators_ = fitted
        self.named_estimators_ = named
        return self

    def _align_proba(self, estimator, proba: np.ndarray) -> np.ndarray:
        proba = np.asarray(proba, dtype=np.float64)
        if proba.ndim != 2:
            raise ValueError("predict_proba must return a 2D array")
        n_classes = int(self.classes_.shape[0])
        est_classes = getattr(estimator, "classes_", None)
        if est_classes is None:
            if proba.shape[1] != n_classes:
                raise ValueError(
                    "estimator without classes_ returned "
                    f"{proba.shape[1]} probability columns, expected {n_classes}"
                )
            return proba
        est_classes = np.asarray(est_classes)
        if est_classes.shape[0] != proba.shape[1]:
            raise ValueError("predict_proba columns do not match estimator.classes_")
        if est_classes.shape[0] == n_classes and np.array_equal(est_classes, self.classes_):
            return proba
        aligned = np.zeros((proba.shape[0], n_classes), dtype=np.float64)
        index = {label: i for i, label in enumerate(self.classes_.tolist())}
        for column, label in enumerate(est_classes.tolist()):
            try:
                aligned[:, index[label]] = proba[:, column]
            except KeyError as exc:
                raise ValueError(
                    f"estimator classes_ contain {label!r}, which was not in the training labels"
                ) from exc
        return aligned

    def predict_proba(self, X) -> np.ndarray:
        """Average class probabilities. Only available when ``voting='soft'``."""
        self._check_fitted()
        if self.voting != "soft":
            raise AttributeError(
                f"predict_proba is not available when voting={self.voting!r}"
            )
        X = self._validate_X(X)
        aligned = [
            self._align_proba(est, est.predict_proba(X)) for est in self.estimators_
        ]
        return np.average(np.stack(aligned, axis=0), axis=0, weights=self._weights_vec())

    def predict(self, X) -> np.ndarray:
        """Predict class labels for ``X``."""
        self._check_fitted()
        X = self._validate_X(X)
        if self.voting == "soft":
            return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

        weights = self._weights_vec()
        label_to_idx = {label: i for i, label in enumerate(self.classes_.tolist())}
        n_samples = X.shape[0]
        scores = np.zeros((n_samples, self.n_classes_), dtype=np.float64)
        for weight, est in zip(weights, self.estimators_):
            pred = np.asarray(est.predict(X)).reshape(-1)
            if pred.shape[0] != n_samples:
                raise ValueError("base estimator predict returned the wrong number of rows")
            encoded = np.empty(n_samples, dtype=np.intp)
            for i, label in enumerate(pred.tolist()):
                try:
                    encoded[i] = label_to_idx[label]
                except KeyError as exc:
                    raise ValueError(
                        f"estimator predicted label {label!r}, which is not in classes_"
                    ) from exc
            scores[np.arange(n_samples), encoded] += weight
        return self.classes_[np.argmax(scores, axis=1)]

    def transform(self, X) -> np.ndarray:
        """Return each base estimator's labels or aligned probabilities.

        Hard voting returns predicted labels of shape
        ``(n_samples, n_estimators)``. Soft voting returns probabilities of
        shape ``(n_samples, n_estimators * n_classes)`` when
        ``flatten_transform`` is true, and
        ``(n_estimators, n_samples, n_classes)`` otherwise. Probability
        columns follow :attr:`classes_`.
        """
        self._check_fitted()
        X = self._validate_X(X)
        if self.voting == "soft":
            probas = np.stack(
                [self._align_proba(est, est.predict_proba(X)) for est in self.estimators_],
                axis=0,
            )
            if not self.flatten_transform:
                return probas
            return np.hstack(probas)
        labels = [np.asarray(est.predict(X)).reshape(-1) for est in self.estimators_]
        return np.column_stack(labels)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Names of the columns produced by :meth:`transform`.

        ``input_features`` is ignored; it exists so the method matches the
        usual scikit-learn signature.
        """
        del input_features
        self._check_fitted()
        if self.voting == "soft" and not self.flatten_transform:
            raise ValueError(
                "get_feature_names_out is not supported when voting='soft' and "
                "flatten_transform=False"
            )
        prefix = type(self).__name__.lower()
        active_names = [name for name, est in self.estimators if est != "drop"]
        if self.voting == "hard":
            names = [f"{prefix}_{name}" for name in active_names]
        else:
            names = [
                f"{prefix}_{name}{class_index}"
                for name in active_names
                for class_index in range(self.n_classes_)
            ]
        return np.asarray(names, dtype=object)
