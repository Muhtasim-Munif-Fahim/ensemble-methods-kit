"""Tests for the EM-trained mixture of experts regressor."""

from __future__ import annotations

import numpy as np
import pytest

from ensemble_methods_kit import MixtureOfExpertsRegressor, r2_score


def _piecewise(n=600, noise=0.2, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-3, 3, n)
    y = np.where(x < 0, 2 * x, -3 * x + 1) + rng.normal(0, noise, n)
    return x[:, None], y


def test_recovers_piecewise_linear_regimes():
    X, y = _piecewise()
    m = MixtureOfExpertsRegressor(n_experts=2, random_state=0).fit(X, y)
    assert m.score(X, y) > 0.98
    single = MixtureOfExpertsRegressor(n_experts=1).fit(X, y)
    assert single.score(X, y) < 0.2
    # slopes of the two experts in original units are {2, -3}
    slopes = sorted(m.coef_[0] / m.x_scale_[0])
    assert slopes == pytest.approx([-3, 2], abs=0.1)
    assert np.sqrt(m.variances_) == pytest.approx([0.2, 0.2], abs=0.03)
    G = m.predict_gates(np.array([[-2.0], [2.0]]))
    assert G.argmax(axis=1)[0] != G.argmax(axis=1)[1]
    assert G.max(axis=1).min() > 0.95


def test_em_objective_is_monotone():
    X, y = _piecewise(400, seed=3)
    for k in (2, 3, 4):
        m = MixtureOfExpertsRegressor(n_experts=k, max_iter=80, n_init=1, random_state=1).fit(X, y)
        assert np.all(np.diff(m.log_likelihood_) > -1e-6 * len(y))


def test_single_expert_is_ridge_regression(regression):
    X, y = regression
    m = MixtureOfExpertsRegressor(n_experts=1, alpha=0.0).fit(X, y)
    A = np.c_[X, np.ones(len(X))]
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    assert np.allclose(m.predict(X), A @ beta, atol=1e-8)
    assert m.variances_[0] == pytest.approx(np.mean((y - A @ beta) ** 2), rel=1e-8)
    assert np.allclose(m.predict_gates(X), 1.0)


def test_probabilistic_outputs_are_consistent():
    X, y = _piecewise(500, seed=1)
    m = MixtureOfExpertsRegressor(n_experts=2, random_state=0).fit(X, y)
    G = m.predict_gates(X)
    assert np.allclose(G.sum(axis=1), 1.0) and np.all(G >= 0)
    R = m.responsibilities(X, y)
    assert np.allclose(R.sum(axis=1), 1.0)
    pred = m.predict(X)
    assert np.allclose(pred, np.sum(G * m.predict_expert_means(X), axis=1))
    std = m.predict_std(X)
    assert np.all(std >= np.sqrt(m.variances_.min()) - 1e-12)
    # final objective equals total log density minus penalty
    ll = m.score_samples(X, y).sum()
    pen = 0.5 * m.alpha * np.sum(m.coef_[:-1] ** 2) + 0.5 * m.gate_alpha * np.sum(m.gate_coef_[:-1] ** 2)
    assert ll - pen == pytest.approx(m.log_likelihood_[-1], rel=1e-6)


def test_input_dependent_noise_is_modelled():
    rng = np.random.default_rng(4)
    x = rng.uniform(-2, 2, 800)
    y = 0.5 * x + np.where(x < 0, 0.1, 1.5) * rng.normal(size=800)
    X = x[:, None]
    m = MixtureOfExpertsRegressor(n_experts=2, random_state=0).fit(X, y)
    std = m.predict_std(np.array([[-1.5], [1.5]]))
    assert std[0] < 0.3 and std[1] > 1.0
    single = MixtureOfExpertsRegressor(n_experts=1).fit(X, y)
    assert m.score_samples(X, y).mean() > single.score_samples(X, y).mean() + 0.2


def test_generalises_and_is_reproducible():
    X, y = _piecewise(800, seed=5)
    Xt, yt = _piecewise(400, seed=6)
    a = MixtureOfExpertsRegressor(n_experts=3, random_state=7).fit(X, y)
    b = MixtureOfExpertsRegressor(n_experts=3, random_state=7).fit(X, y)
    assert np.allclose(a.predict(Xt), b.predict(Xt))
    assert r2_score(yt, a.predict(Xt)) > 0.97


def test_validation():
    with pytest.raises(ValueError):
        MixtureOfExpertsRegressor(n_experts=0)
    with pytest.raises(ValueError):
        MixtureOfExpertsRegressor(alpha=-1)
    with pytest.raises(ValueError):
        MixtureOfExpertsRegressor(min_variance=0)
    m = MixtureOfExpertsRegressor()
    with pytest.raises(RuntimeError):
        m.predict(np.zeros((2, 1)))
    with pytest.raises(ValueError):
        m.fit(np.zeros((3, 1)), np.zeros(2))
    X, y = _piecewise(50)
    m.fit(X, y)
    with pytest.raises(ValueError):
        m.predict(np.zeros((2, 3)))
