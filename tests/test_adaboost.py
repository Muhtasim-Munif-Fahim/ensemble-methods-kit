"""Tests for AdaBoostClassifier."""
from __future__ import annotations

import numpy as np

import pytest

from ensemble_methods_kit import AdaBoostClassifier, accuracy_score


def _linear_separable() -> tuple:
    rng = np.random.default_rng(42)
    n = 50
    X0 = rng.normal(0, 1, size=(n, 2))
    X1 = rng.normal(5, 1, size=(n, 2))
    X = np.vstack([X0, X1])
    y = np.array([0] * n + [1] * n)
    return X, y


def test_fit_predict_basic() -> None:
    X, y = _linear_separable()
    clf = AdaBoostClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    preds = clf.predict(X)
    acc = accuracy_score(y, preds)
    assert acc > 0.9


def test_predict_before_fit() -> None:
    clf = AdaBoostClassifier(n_estimators=10)
    with pytest.raises(RuntimeError, match="not fitted"):
        clf.predict(np.array([[1.0, 2.0]]))


def test_predict_proba_shape() -> None:
    X, y = _linear_separable()
    clf = AdaBoostClassifier(n_estimators=5, random_state=42)
    clf.fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (100, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_multiclass_raises() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 1, 2, 3])
    clf = AdaBoostClassifier(n_estimators=5)
    with pytest.raises(ValueError, match="binary"):
        clf.fit(X, y)


def test_n_estimators_validation() -> None:
    with pytest.raises(ValueError, match="n_estimators"):
        AdaBoostClassifier(n_estimators=0)


def test_learning_rate_validation() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        AdaBoostClassifier(learning_rate=-1.0)


def test_weights_length() -> None:
    X, y = _linear_separable()
    clf = AdaBoostClassifier(n_estimators=5, random_state=42)
    clf.fit(X, y)
    assert len(clf.estimator_weights_) == len(clf.estimators_)
    assert all(w > 0 for w in clf.weights_)


def test_reproducibility() -> None:
    X, y = _linear_separable()
    clf1 = AdaBoostClassifier(n_estimators=10, random_state=42)
    clf1.fit(X, y)
    clf2 = AdaBoostClassifier(n_estimators=10, random_state=42)
    clf2.fit(X, y)
    preds1 = clf1.predict(X)
    preds2 = clf2.predict(X)
    np.testing.assert_array_equal(preds1, preds2)


def test_feature_importances_shape() -> None:
    X, y = _linear_separable()
    clf = AdaBoostClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    importances = clf.feature_importances_
    assert importances.shape == (2,)
    assert abs(importances.sum() - 1.0) < 1e-6


def test_feature_importances_predicts_informative_feature() -> None:
    X, y = _linear_separable()
    clf = AdaBoostClassifier(n_estimators=20, random_state=0)
    clf.fit(X, y)
    importances = clf.feature_importances_
    assert importances[0] > importances[1]


def test_feature_importances_before_fit_raises() -> None:
    clf = AdaBoostClassifier(n_estimators=5)
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = clf.feature_importances_
