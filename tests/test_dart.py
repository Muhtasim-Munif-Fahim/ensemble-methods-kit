"""Tests for DART (dropout) gradient boosting."""

from __future__ import annotations

import numpy as np
import pytest

from ensemble_methods_kit import (
    DARTClassifier,
    DARTRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    accuracy_score,
    r2_score,
)
from ensemble_methods_kit.utils import train_test_split


def test_rate_drop_zero_matches_gradient_boosting_regressor(regression):
    X, y = regression
    kw = dict(n_estimators=25, learning_rate=0.2, max_depth=3, subsample=0.8, random_state=3)
    gb = GradientBoostingRegressor(**kw).fit(X, y)
    dart = DARTRegressor(rate_drop=0.0, **kw).fit(X, y)
    np.testing.assert_allclose(dart.predict(X), gb.predict(X), atol=1e-12)
    assert np.all(dart.tree_weights_ == 0.2)


def test_rate_drop_zero_matches_gradient_boosting_classifier(multiclass_cls):
    X, y = multiclass_cls
    kw = dict(n_estimators=10, learning_rate=0.3, max_depth=2, random_state=1)
    gb = GradientBoostingClassifier(**kw).fit(X, y)
    dart = DARTClassifier(rate_drop=0.0, **kw).fit(X, y)
    np.testing.assert_allclose(dart.predict_proba(X), gb.predict_proba(X), atol=1e-12)


def test_dart_regressor_learns_nonlinear_signal():
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, (400, 2))
    y = np.sin(X[:, 0]) * 2 + 0.5 * X[:, 1] ** 2 + rng.normal(0, 0.2, 400)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    m = DARTRegressor(n_estimators=80, learning_rate=0.3, max_depth=3,
                      rate_drop=0.15, random_state=0).fit(Xtr, ytr)
    assert r2_score(yte, m.predict(Xte)) > 0.85
    assert sum(m.n_dropped_) > 0


def test_normalisation_weights():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 2))
    y = X[:, 0]
    m = DARTRegressor(n_estimators=12, learning_rate=0.5, rate_drop=0.5,
                      one_drop=True, random_state=4).fit(X, y)
    # first stage cannot drop anything; afterwards one_drop forces >= 1 drop
    assert m.n_dropped_[0] == 0
    assert all(k >= 1 for k in m.n_dropped_[1:])
    for stage, k in enumerate(m.n_dropped_):
        assert k <= stage
    # every new stage started at weight eta/(k+eta) and only shrank afterwards
    for stage, k in enumerate(m.n_dropped_):
        assert m.tree_weights_[stage] <= 0.5 / (k + 0.5) + 1e-12
    assert np.all(m.tree_weights_ > 0)


def test_skip_drop_one_disables_dropout(regression):
    X, y = regression
    m = DARTRegressor(n_estimators=15, rate_drop=0.9, skip_drop=1.0, random_state=0).fit(X, y)
    assert sum(m.n_dropped_) == 0


def test_dart_classifier_accuracy_and_proba(binary_cls, multiclass_cls):
    for X, y in (binary_cls, multiclass_cls):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
        m = DARTClassifier(n_estimators=30, learning_rate=0.3, max_depth=2,
                           rate_drop=0.2, random_state=0).fit(Xtr, ytr)
        assert accuracy_score(yte, m.predict(Xte)) > 0.9
        P = m.predict_proba(Xte)
        np.testing.assert_allclose(P.sum(axis=1), 1.0)


def test_reproducible_and_validates(regression):
    X, y = regression
    a = DARTRegressor(n_estimators=10, rate_drop=0.3, random_state=7).fit(X, y).predict(X)
    b = DARTRegressor(n_estimators=10, rate_drop=0.3, random_state=7).fit(X, y).predict(X)
    np.testing.assert_array_equal(a, b)
    for bad in (dict(rate_drop=1.5), dict(skip_drop=-0.1), dict(learning_rate=0.0),
                dict(n_estimators=0), dict(subsample=0.0)):
        with pytest.raises(ValueError):
            DARTRegressor(**bad).fit(X, y)
