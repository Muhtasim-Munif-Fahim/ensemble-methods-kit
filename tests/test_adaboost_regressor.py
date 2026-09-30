"""Tests for AdaBoostRegressor (AdaBoost.R2)."""

from __future__ import annotations

import numpy as np
import pytest

from ensemble_methods_kit import AdaBoostRegressor, mean_squared_error, r2_score
from ensemble_methods_kit.utils import train_test_split


def _regression_data(n=200, d=4, seed=0, noise=0.15):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, d))
    w = np.array([1.5, -2.0, 0.75, -0.5][:d], dtype=float)
    y = X @ w + noise * rng.normal(0, 1, n)
    return X, y


def test_fit_predict_shape_and_score():
    X, y = _regression_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    model = AdaBoostRegressor(
        n_estimators=40, learning_rate=0.5, max_depth=3, random_state=0
    ).fit(Xtr, ytr)
    preds = model.predict(Xte)
    assert preds.shape == (len(yte),)
    assert r2_score(yte, preds) > 0.7
    assert mean_squared_error(yte, preds) < mean_squared_error(
        yte, np.full_like(yte, ytr.mean())
    )
    assert len(model.estimators_) >= 1
    assert len(model.estimator_weights_) == len(model.estimators_)
    assert len(model.estimator_errors_) == len(model.estimators_)


def test_more_estimators_helps():
    X, y = _regression_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    small = AdaBoostRegressor(
        n_estimators=5, learning_rate=0.5, max_depth=2, random_state=0
    ).fit(Xtr, ytr)
    large = AdaBoostRegressor(
        n_estimators=40, learning_rate=0.5, max_depth=2, random_state=0
    ).fit(Xtr, ytr)
    assert mean_squared_error(yte, small.predict(Xte)) >= mean_squared_error(
        yte, large.predict(Xte)
    ) - 1e-6


def test_reproducible():
    X, y = _regression_data()
    Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.3, random_state=42)
    kwargs = dict(n_estimators=20, learning_rate=0.5, max_depth=2, random_state=5)
    a = AdaBoostRegressor(**kwargs).fit(Xtr, ytr).predict(Xte)
    b = AdaBoostRegressor(**kwargs).fit(Xtr, ytr).predict(Xte)
    assert np.allclose(a, b)


def test_staged_predict_matches_final():
    X, y = _regression_data(n=120)
    Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.3, random_state=42)
    model = AdaBoostRegressor(
        n_estimators=15, learning_rate=0.5, max_depth=2, random_state=0
    ).fit(Xtr, ytr)
    stages = list(model.staged_predict(Xte))
    assert len(stages) == len(model.estimators_)
    assert np.allclose(stages[-1], model.predict(Xte))


def test_loss_variants_run():
    X, y = _regression_data(n=100)
    for loss in ("linear", "square", "exponential"):
        model = AdaBoostRegressor(
            n_estimators=10, max_depth=2, loss=loss, random_state=0
        ).fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(y),)
        assert np.all(np.isfinite(preds))


def test_feature_importances_rank_signal():
    rng = np.random.default_rng(0)
    signal = rng.normal(size=240)
    X = np.column_stack(
        [signal, rng.normal(size=240), rng.normal(size=240), np.zeros(240)]
    )
    y = 2.0 * signal + 0.1 * rng.normal(size=240)
    model = AdaBoostRegressor(
        n_estimators=30, max_depth=3, learning_rate=0.5, random_state=0
    ).fit(X, y)
    importances = model.feature_importances_
    assert importances.shape == (4,)
    assert importances.min() >= 0.0
    assert importances.sum() == pytest.approx(1.0)
    assert importances[0] == importances.max()


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        AdaBoostRegressor(n_estimators=0)
    with pytest.raises(ValueError):
        AdaBoostRegressor(learning_rate=-1.0)
    with pytest.raises(ValueError):
        AdaBoostRegressor(max_depth=0)
    with pytest.raises(ValueError):
        AdaBoostRegressor(loss="nope")


def test_predict_before_fit_raises():
    model = AdaBoostRegressor(n_estimators=5)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(np.zeros((3, 2)))


def test_export_available():
    from ensemble_methods_kit import AdaBoostRegressor as exported

    assert exported is AdaBoostRegressor
