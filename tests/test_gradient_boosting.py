"""Tests for gradient boosting (regressor and classifier)."""

from __future__ import annotations

import numpy as np

from ensemble_methods_kit import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    accuracy_score,
    log_loss,
    r2_score,
)
from ensemble_methods_kit.utils import train_test_split


def test_gradient_boosting_regressor_fits_linear(regression):
    X, y = regression
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    gbr = GradientBoostingRegressor(
        n_estimators=60, learning_rate=0.1, max_depth=3, random_state=0
    ).fit(Xtr, ytr)
    assert r2_score(yte, gbr.predict(Xte)) > 0.8


def test_gradient_boosting_classifier_proba(binary_cls):
    X, y = binary_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    gbc = GradientBoostingClassifier(
        n_estimators=60, learning_rate=0.1, max_depth=2, random_state=0
    ).fit(Xtr, ytr)
    assert accuracy_score(yte, gbc.predict(Xte)) > 0.9
    proba = gbc.predict_proba(Xte)
    assert proba.shape == (len(yte), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert log_loss(yte, proba) < 0.4


def test_gradient_boosting_multiclass(multiclass_cls):
    X, y = multiclass_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    gbc = GradientBoostingClassifier(
        n_estimators=40, learning_rate=0.1, max_depth=2, random_state=1
    ).fit(Xtr, ytr)
    assert gbc.classes_.tolist() == [0, 1, 2]
    assert accuracy_score(yte, gbc.predict(Xte)) > 0.85


def test_gradient_boosting_more_estimators_reduces_loss(binary_cls):
    X, y = binary_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    small = GradientBoostingClassifier(
        n_estimators=10, learning_rate=0.1, max_depth=2, random_state=0
    ).fit(Xtr, ytr)
    large = GradientBoostingClassifier(
        n_estimators=50, learning_rate=0.1, max_depth=2, random_state=0
    ).fit(Xtr, ytr)
    assert log_loss(yte, small.predict_proba(Xte)) >= log_loss(yte, large.predict_proba(Xte))


def test_gradient_boosting_reproducible(binary_cls):
    X, y = binary_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    a = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_depth=2,
                                   random_state=5).fit(Xtr, ytr).predict(Xte)
    b = GradientBoostingClassifier(n_estimators=20, learning_rate=0.1, max_depth=2,
                                   random_state=5).fit(Xtr, ytr).predict(Xte)
    assert np.array_equal(a, b)
