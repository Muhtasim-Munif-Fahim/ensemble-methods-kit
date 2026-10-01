"""Tests for BlendingRegressor."""

from __future__ import annotations

import numpy as np
import pytest

from ensemble_methods_kit import (
    BlendingRegressor,
    DecisionTree,
    GradientBoostingRegressor,
    RandomForestRegressor,
    RidgeRegression,
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


def test_blending_regressor_predict_shape_and_score():
    X, y = _regression_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    blend = BlendingRegressor(_bases(), validation_fraction=0.3, random_state=0).fit(Xtr, ytr)
    preds = blend.predict(Xte)
    assert preds.shape == (len(yte),)
    assert r2_score(yte, preds) > 0.65
    assert mean_squared_error(yte, preds) < mean_squared_error(yte, np.full_like(yte, ytr.mean()))


def test_blending_regressor_beats_weak_base():
    X, y = _regression_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    weak = DecisionTree(criterion="variance", max_depth=1, random_state=0)
    blend = BlendingRegressor([("weak", weak)], validation_fraction=0.3, random_state=0).fit(Xtr, ytr)
    base_r2 = r2_score(yte, weak.fit(Xtr, ytr).predict(Xte))
    blend_r2 = r2_score(yte, blend.predict(Xte))
    assert blend_r2 >= base_r2 - 0.1


def test_blending_regressor_passthrough():
    X, y = _regression_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    blend = BlendingRegressor(_bases(), validation_fraction=0.3, passthrough=True, random_state=0).fit(Xtr, ytr)
    assert blend.predict(Xte).shape == (len(yte),)
    assert r2_score(yte, blend.predict(Xte)) > 0.65


def test_blending_regressor_reproducible():
    X, y = _regression_data()
    Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.3, random_state=42)
    a = BlendingRegressor(_bases(), validation_fraction=0.3, random_state=0).fit(Xtr, ytr).predict(Xte)
    b = BlendingRegressor(_bases(), validation_fraction=0.3, random_state=0).fit(Xtr, ytr).predict(Xte)
    assert np.allclose(a, b)


def test_blending_regressor_custom_meta():
    X, y = _regression_data(n=120)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    blend = BlendingRegressor(
        _bases(),
        meta_estimator=RidgeRegression(alpha=0.1),
        validation_fraction=0.3,
        random_state=0,
    ).fit(Xtr, ytr)
    assert r2_score(yte, blend.predict(Xte)) > 0.6


def test_exports():
    from ensemble_methods_kit import BlendingRegressor as B

    assert B is BlendingRegressor
