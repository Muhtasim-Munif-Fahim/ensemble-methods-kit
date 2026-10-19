"""Tests for Meinshausen quantile regression forests and DecisionTree.apply."""

from __future__ import annotations

import numpy as np
import pytest

from ensemble_methods_kit import (
    DecisionTree,
    QuantileRegressionForest,
    RandomForestRegressor,
    interval_coverage,
    pinball_loss,
)


def _hetero(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 10.0, (n, 3))
    y = np.sin(X[:, 0]) + (0.1 + 0.3 * X[:, 0]) * rng.normal(size=n)
    return X, y


@pytest.fixture(scope="module")
def fitted():
    X, y = _hetero(600, 0)
    model = QuantileRegressionForest(n_estimators=60, random_state=0).fit(X, y)
    return model, X, y


def test_tree_apply_numbers_leaves_consistently():
    X, y = _hetero(200, 1)
    tree = DecisionTree(criterion="variance", max_depth=4, random_state=0).fit(X, y)
    leaves = tree.apply(X)
    assert leaves.min() == 0 and leaves.max() == tree.n_leaves_ - 1
    preds = tree.predict(X)
    for leaf in np.unique(leaves):
        assert np.ptp(preds[leaves == leaf]) == 0.0
        # The leaf value is the mean of its training targets.
        assert preds[leaves == leaf][0] == pytest.approx(y[leaves == leaf].mean())
    assert np.array_equal(tree.apply(X), leaves)
    with pytest.raises(RuntimeError):
        DecisionTree(criterion="variance").apply(X)


def test_weights_form_a_distribution(fitted):
    model, X, _ = fitted
    Xq, _ = _hetero(30, 2)
    W = model.conditional_weights(Xq)
    assert W.shape == (30, X.shape[0])
    assert np.all(W >= 0)
    np.testing.assert_allclose(W.sum(axis=1), 1.0)


def test_inbag_weighting_reproduces_the_forest_mean():
    X, y = _hetero(300, 3)
    model = QuantileRegressionForest(n_estimators=25, leaf_weighting="inbag", random_state=1).fit(X, y)
    Xq, _ = _hetero(40, 4)
    np.testing.assert_allclose(model.conditional_weights(Xq) @ y, model.predict(Xq), atol=1e-10)


def test_single_tree_quantiles_are_leaf_order_statistics():
    X, y = _hetero(150, 5)
    model = QuantileRegressionForest(n_estimators=1, bootstrap=False, max_features=None, min_samples_split=20, random_state=0).fit(X, y)
    tree = model.estimators_[0]
    Xq = X[:10]
    q = model.predict_quantiles(Xq, [0.0, 0.5, 1.0])
    train_leaves = tree.apply(X)
    for row, leaf in zip(q, tree.apply(Xq)):
        members = np.sort(y[train_leaves == leaf])
        assert row[0] == members[0]
        assert row[2] == members[-1]
        assert row[1] == members[int(np.ceil(0.5 * members.size)) - 1]


def test_intervals_cover_and_adapt_to_heteroscedastic_noise(fitted):
    model, _, _ = fitted
    Xt, yt = _hetero(800, 6)
    lo, hi = model.predict_interval(Xt, coverage=0.9)
    assert 0.83 <= interval_coverage(yt, lo, hi) <= 0.96
    width = hi - lo
    assert width[Xt[:, 0] > 8].mean() > 2.5 * width[Xt[:, 0] < 2].mean()
    q = model.predict_quantiles(Xt[:50], [0.1, 0.25, 0.5, 0.75, 0.9])
    assert np.all(np.diff(q, axis=1) >= 0)


def test_median_beats_mean_under_pinball_loss_on_skewed_noise():
    rng = np.random.default_rng(7)
    X = rng.uniform(0, 1, (700, 2))
    y = X[:, 0] + rng.exponential(1.0, 700)
    model = QuantileRegressionForest(n_estimators=60, random_state=2).fit(X[:500], y[:500])
    med = model.predict_quantiles(X[500:], 0.5)
    mean = model.predict(X[500:])
    assert med.shape == (200,)
    assert pinball_loss(y[500:], med, 0.5) < pinball_loss(y[500:], mean, 0.5)
    q90 = model.predict_quantiles(X[500:], 0.9)
    assert pinball_loss(y[500:], q90, 0.9) < pinball_loss(y[500:], mean, 0.9)


def test_oob_quantiles_are_honest(fitted):
    model, X, y = fitted
    oob = model.oob_predict_quantiles([0.05, 0.95])
    valid = ~np.isnan(oob).any(axis=1)
    assert valid.mean() > 0.99
    cov = interval_coverage(y[valid], oob[valid, 0], oob[valid, 1])
    assert 0.83 <= cov <= 0.96
    # In-sample intervals (own target included) are optimistic by comparison.
    lo, hi = model.predict_interval(X, 0.9)
    assert interval_coverage(y, lo, hi) >= cov


def test_oob_score_uses_weighted_oob_mean():
    X, y = _hetero(300, 8)
    model = QuantileRegressionForest(n_estimators=40, oob_score=True, random_state=3).fit(X, y)
    assert 0.0 < model.oob_score_ < 1.0
    assert model.oob_prediction_.shape == (300,)
    rf = RandomForestRegressor(n_estimators=40, max_depth=None, min_samples_split=10, oob_score=True, random_state=3).fit(X, y)
    assert model.oob_score_ == pytest.approx(rf.oob_score_, abs=0.15)


def test_pinball_and_coverage_metrics():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([2.0, 2.0, 2.0])
    # q=0.9: losses 0.1, 0, 0.9 -> mean 1/3
    assert pinball_loss(y, p, 0.9) == pytest.approx(1.0 / 3.0)
    assert pinball_loss(y, p, 0.5) == pytest.approx(1.0 / 3.0)
    assert interval_coverage(y, [0, 2.5, 2.9], [1, 3, 3.1]) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        pinball_loss(y, p, 1.0)
    with pytest.raises(ValueError):
        interval_coverage(y, [0, 1], [1, 2])


def test_validation(fitted):
    model, X, y = fitted
    with pytest.raises(ValueError, match="leaf_weighting"):
        QuantileRegressionForest(leaf_weighting="mean")
    with pytest.raises(ValueError, match="chunk_size"):
        QuantileRegressionForest(chunk_size=0)
    with pytest.raises(ValueError, match="quantiles"):
        model.predict_quantiles(X[:2], [1.5])
    with pytest.raises(ValueError, match="coverage"):
        model.predict_interval(X[:2], 1.0)
    with pytest.raises(ValueError, match="features"):
        model.predict_quantiles(X[:2, :2], 0.5)
    with pytest.raises(RuntimeError):
        QuantileRegressionForest().predict_quantiles(X[:2], 0.5)
    no_boot = QuantileRegressionForest(n_estimators=3, bootstrap=False, random_state=0).fit(X[:50], y[:50])
    with pytest.raises(ValueError, match="bootstrap"):
        no_boot.oob_predict_quantiles()
