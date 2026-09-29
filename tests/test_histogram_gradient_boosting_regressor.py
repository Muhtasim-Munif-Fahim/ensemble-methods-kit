"""Tests for HistogramGradientBoostingRegressor."""

from __future__ import annotations

import numpy as np
import pytest

from ensemble_methods_kit import (
    HistogramGradientBoostingRegressor,
    mean_squared_error,
    r2_score,
)
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
    model = HistogramGradientBoostingRegressor(
        n_estimators=40, learning_rate=0.1, max_depth=3, random_state=0
    ).fit(Xtr, ytr)
    preds = model.predict(Xte)
    assert preds.shape == (len(yte),)
    assert r2_score(yte, preds) > 0.85
    assert mean_squared_error(yte, preds) < mean_squared_error(
        yte, np.full_like(yte, ytr.mean())
    )
    assert model.n_iter_ == 40
    assert len(model.estimators_) == 40
    assert len(model.train_score_) == 40
    assert model.train_score_[-1] <= model.train_score_[0] + 1e-9


def test_staged_predict_matches_final():
    X, y = _regression_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    model = HistogramGradientBoostingRegressor(
        n_estimators=15, learning_rate=0.1, max_depth=2, random_state=0
    ).fit(Xtr, ytr)
    stages = list(model.staged_predict(Xte))
    assert len(stages) == model.n_estimators
    assert np.allclose(stages[-1], model.predict(Xte))
    assert mean_squared_error(yte, stages[0]) >= mean_squared_error(yte, stages[-1]) - 1e-9


def test_more_estimators_reduces_mse():
    X, y = _regression_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    small = HistogramGradientBoostingRegressor(
        n_estimators=5, learning_rate=0.1, max_depth=2, random_state=0
    ).fit(Xtr, ytr)
    large = HistogramGradientBoostingRegressor(
        n_estimators=40, learning_rate=0.1, max_depth=2, random_state=0
    ).fit(Xtr, ytr)
    assert mean_squared_error(yte, small.predict(Xte)) >= mean_squared_error(
        yte, large.predict(Xte)
    )


def test_reproducible_with_subsampling():
    X, y = _regression_data()
    Xtr, Xte, ytr, _ = train_test_split(X, y, test_size=0.3, random_state=42)
    kwargs = dict(
        n_estimators=15, learning_rate=0.1, max_depth=2, subsample=0.7, random_state=5
    )
    a = HistogramGradientBoostingRegressor(**kwargs).fit(Xtr, ytr).predict(Xte)
    b = HistogramGradientBoostingRegressor(**kwargs).fit(Xtr, ytr).predict(Xte)
    assert np.allclose(a, b)


def test_feature_importances_rank_signal():
    rng = np.random.default_rng(0)
    signal = rng.normal(size=240)
    X = np.column_stack(
        [signal, rng.normal(size=240), rng.normal(size=240), np.zeros(240)]
    )
    y = 2.0 * signal + 0.1 * rng.normal(size=240)
    model = HistogramGradientBoostingRegressor(
        n_estimators=25, max_depth=3, random_state=0
    ).fit(X, y)
    importances = model.feature_importances_
    assert importances.shape == (4,)
    assert importances.min() >= 0.0
    assert importances.sum() == pytest.approx(1.0)
    assert importances[0] == importances.max()
    assert importances[3] == 0.0


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        HistogramGradientBoostingRegressor(n_estimators=0)
    with pytest.raises(ValueError):
        HistogramGradientBoostingRegressor(learning_rate=0.0)
    with pytest.raises(ValueError):
        HistogramGradientBoostingRegressor(max_depth=0)
    with pytest.raises(ValueError):
        HistogramGradientBoostingRegressor(max_bins=1)
    with pytest.raises(ValueError):
        HistogramGradientBoostingRegressor(min_samples_leaf=0)
    with pytest.raises(ValueError):
        HistogramGradientBoostingRegressor(l2_regularization=-0.1)
    with pytest.raises(ValueError):
        HistogramGradientBoostingRegressor(subsample=0.0)


def test_predict_before_fit_raises():
    model = HistogramGradientBoostingRegressor(n_estimators=2)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict([[0.0, 1.0]])


def test_feature_mismatch_raises():
    X, y = _regression_data(n=40, d=3)
    model = HistogramGradientBoostingRegressor(n_estimators=3, random_state=0).fit(X, y)
    with pytest.raises(ValueError, match="features"):
        model.predict(np.zeros((5, 2)))


def test_exported_from_public_api():
    from ensemble_methods_kit.histogram_gradient_boosting import (
        HistogramGradientBoostingRegressor as impl,
    )

    assert HistogramGradientBoostingRegressor is impl
