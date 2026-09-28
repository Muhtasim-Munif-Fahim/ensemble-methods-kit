"""Tests for StackingRegressor and its RidgeRegression meta-learner."""

from __future__ import annotations

import numpy as np
import pytest

from ensemble_methods_kit import (
    DecisionTree,
    GradientBoostingRegressor,
    RandomForestRegressor,
    RidgeRegression,
    StackingRegressor,
    mean_squared_error,
    r2_score,
)
from ensemble_methods_kit.utils import train_test_split


def _regression_data(n=180, d=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, d))
    w = np.array([1.5, -2.0, 0.5][:d], dtype=float)
    y = X @ w + 0.15 * rng.normal(0, 1, n)
    return X, y


def _bases():
    return [
        ("rf", RandomForestRegressor(n_estimators=12, max_depth=4, random_state=1)),
        ("gbr", GradientBoostingRegressor(n_estimators=25, learning_rate=0.1,
                                          max_depth=2, random_state=1)),
        ("dt", DecisionTree(criterion="variance", max_depth=4, random_state=1)),
    ]


def test_ridge_recovers_linear_target():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (200, 2))
    y = 3.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5
    ridge = RidgeRegression(alpha=0.01).fit(X, y)
    preds = ridge.predict(X)
    assert r2_score(y, preds) > 0.99


def test_stacking_regressor_predict_shape_and_score():
    X, y = _regression_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    stack = StackingRegressor(_bases(), cv=3, random_state=0).fit(Xtr, ytr)
    preds = stack.predict(Xte)
    assert preds.shape == (len(yte),)
    assert r2_score(yte, preds) > 0.7
    assert mean_squared_error(yte, preds) < mean_squared_error(yte, np.full_like(yte, ytr.mean()))


def test_stacking_regressor_beats_weak_base():
    X, y = _regression_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    weak = DecisionTree(criterion="variance", max_depth=1, random_state=0)
    stack = StackingRegressor([("weak", weak)], cv=3, random_state=0).fit(Xtr, ytr)
    base_r2 = r2_score(yte, weak.fit(Xtr, ytr).predict(Xte))
    stack_r2 = r2_score(yte, stack.predict(Xte))
    assert stack_r2 >= base_r2 - 0.05


def test_stacking_regressor_passthrough():
    X, y = _regression_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    stack = StackingRegressor(_bases(), cv=3, passthrough=True, random_state=0).fit(Xtr, ytr)
    assert stack.predict(Xte).shape == (len(yte),)
    assert r2_score(yte, stack.predict(Xte)) > 0.7


def test_stacking_regressor_reproducible():
    X, y = _regression_data()
    Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.3, random_state=42)
    a = StackingRegressor(_bases(), cv=3, random_state=0).fit(Xtr, ytr).predict(Xte)
    b = StackingRegressor(_bases(), cv=3, random_state=0).fit(Xtr, ytr).predict(Xte)
    assert np.allclose(a, b)


def test_stacking_regressor_custom_meta():
    X, y = _regression_data(n=120)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    stack = StackingRegressor(
        _bases(),
        meta_estimator=RidgeRegression(alpha=0.1),
        cv=3,
        random_state=0,
    ).fit(Xtr, ytr)
    assert r2_score(yte, stack.predict(Xte)) > 0.65


def test_exports():
    from ensemble_methods_kit import RidgeRegression as R
    from ensemble_methods_kit import StackingRegressor as S

    assert R is RidgeRegression
    assert S is StackingRegressor
