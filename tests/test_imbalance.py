"""Tests for EasyEnsemble and BalanceCascade."""

from __future__ import annotations

import numpy as np
import pytest

from ensemble_methods_kit import (
    BalanceCascadeClassifier,
    EasyEnsembleClassifier,
    accuracy_score,
)


def _imbalanced_blobs(seed=0, n_maj=200, n_min=20, d=4):
    rng = np.random.default_rng(seed)
    X_maj = rng.normal(0.0, 1.0, size=(n_maj, d))
    X_min = rng.normal(2.5, 1.0, size=(n_min, d))
    X = np.vstack([X_maj, X_min])
    y = np.concatenate([np.zeros(n_maj, dtype=int), np.ones(n_min, dtype=int)])
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def test_easy_ensemble_improves_minority_recall():
    X, y = _imbalanced_blobs()
    # hold-out last 40
    Xtr, ytr = X[:-40], y[:-40]
    Xte, yte = X[-40:], y[-40:]
    clf = EasyEnsembleClassifier(n_subsets=5, n_estimators=5, max_depth=4, random_state=0)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    # minority class = 1
    min_mask = yte == 1
    if min_mask.any():
        recall = (pred[min_mask] == 1).mean()
        assert recall >= 0.3  # should catch some minority
    assert accuracy_score(yte, pred) > 0.5
    assert clf.classes_.tolist() == [0, 1]
    assert len(clf.estimators_) == 5


def test_easy_ensemble_proba_sums_to_one():
    X, y = _imbalanced_blobs(seed=1)
    clf = EasyEnsembleClassifier(n_subsets=3, n_estimators=3, random_state=1).fit(X, y)
    proba = clf.predict_proba(X[:10])
    assert proba.shape == (10, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert (proba >= 0).all()


def test_balance_cascade_fits_and_predicts():
    X, y = _imbalanced_blobs(seed=2, n_maj=150, n_min=25)
    clf = BalanceCascadeClassifier(
        n_subsets=4, n_estimators=4, removal_threshold=0.6, random_state=2
    )
    clf.fit(X, y)
    pred = clf.predict(X)
    assert pred.shape == y.shape
    assert set(np.unique(pred)).issubset({0, 1})
    assert len(clf.estimators_) >= 1


def test_easy_ensemble_rejects_single_class():
    X = np.random.default_rng(0).normal(size=(20, 3))
    y = np.zeros(20, dtype=int)
    with pytest.raises(ValueError, match="two classes"):
        EasyEnsembleClassifier(n_subsets=2, n_estimators=2, random_state=0).fit(X, y)


def test_balance_cascade_invalid_threshold():
    with pytest.raises(ValueError, match="removal_threshold"):
        BalanceCascadeClassifier(removal_threshold=1.0)
