"""Voting ensembles that combine independent classifiers."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["VotingClassifier"]


class VotingClassifier:
    """Combine multiple classifiers via soft or hard voting.

    Parameters
    ----------
    estimators :
        List of ``(name, estimator)`` pairs.  Each estimator must expose
        ``fit`` and (``predict`` or ``predict_proba``).
    voting :
        ``"soft"`` averages the class probabilities; ``"hard"`` takes a
        majority vote of the predicted labels.
    weights :
        Weight of each estimator.  Defaults to equal weights.
    """

    def __init__(
        self,
        estimators: Sequence[Tuple[str, object]],
        voting: str = "soft",
        weights: Optional[Sequence[float]] = None,
        n_jobs: Optional[int] = None,
    ) -> None:
        if voting not in ("soft", "hard"):
            raise ValueError("voting must be 'soft' or 'hard'")
        self.estimators = list(estimators)
        self.voting = voting
        self.weights = weights
        self.n_jobs = n_jobs
        self.named_estimators_: dict = {}
        self.classes_: Optional[np.ndarray] = None

    def _weights_vec(self) -> np.ndarray:
        if self.weights is None:
            return np.ones(len(self.estimators))
        return np.asarray(self.weights, dtype=np.float64)

    def fit(self, X, y) -> "VotingClassifier":
        self.named_estimators_ = {}
        for name, est in self.estimators:
            est.fit(X, y)
            self.named_estimators_[name] = est
        self.classes_ = np.unique(np.asarray(y))
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.voting != "soft":
            raise AttributeError("predict_proba is only available for soft voting")
        probas = np.array([est.predict_proba(X) for _, est in self.estimators])
        w = self._weights_vec()
        return np.average(probas, axis=0, weights=w)

    def predict(self, X) -> np.ndarray:
        if self.voting == "soft":
            proba = self.predict_proba(X)
            return self.classes_[proba.argmax(axis=1)]
        preds = np.array([est.predict(X) for _, est in self.estimators])
        w = self._weights_vec()
        n_samples = preds.shape[1]
        out = np.empty(n_samples, dtype=self.classes_.dtype)
        for i in range(n_samples):
            idx = np.searchsorted(self.classes_, preds[:, i])
            scores = np.zeros(self.classes_.shape[0])
            np.add.at(scores, idx, w)
            out[i] = self.classes_[scores.argmax()]
        return out

    @property
    def estimators_(self) -> List:
        return list(self.named_estimators_.values())
