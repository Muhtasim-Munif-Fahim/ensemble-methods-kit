"""Tests for the Isolation Forest anomaly detector."""

from __future__ import annotations

import numpy as np
import pytest

from ensemble_methods_kit import IsolationForest, average_path_length, roc_auc_score


def _data(seed=0):
    rng = np.random.default_rng(seed)
    inliers = rng.normal(0.0, 1.0, (300, 2))
    outliers = rng.uniform(6.0, 9.0, (15, 2)) * rng.choice([-1, 1], (15, 2))
    X = np.vstack([inliers, outliers])
    y = np.r_[np.zeros(300, dtype=int), np.ones(15, dtype=int)]
    return X, y


def test_average_path_length_values():
    np.testing.assert_allclose(average_path_length([0, 1, 2]), [0.0, 0.0, 1.0])
    c256 = float(average_path_length(256))
    expected = 2 * (np.log(255) + 0.5772156649015329) - 2 * 255 / 256
    assert c256 == pytest.approx(expected)


def test_detects_planted_outliers():
    X, y = _data()
    model = IsolationForest(n_estimators=100, random_state=0).fit(X)
    scores = model.score_samples(X)
    assert scores.shape == (315,)
    assert np.all((scores > 0) & (scores <= 1))
    assert roc_auc_score(y, scores) > 0.98
    assert scores[y == 1].mean() > 0.6 > scores[y == 0].mean()


def test_contamination_threshold_flags_expected_fraction():
    X, y = _data(1)
    model = IsolationForest(contamination=0.05, random_state=0)
    pred = model.fit_predict(X)
    assert set(np.unique(pred)) <= {0, 1}
    assert abs(pred.mean() - 0.05) < 0.01
    assert pred[y == 1].mean() > 0.9
    np.testing.assert_array_equal(model.decision_function(X) > 0, pred == 1)


def test_reproducible_and_float_max_samples():
    X, _ = _data(2)
    a = IsolationForest(n_estimators=20, max_samples=0.5, random_state=7).fit(X)
    b = IsolationForest(n_estimators=20, max_samples=0.5, random_state=7).fit(X)
    assert a.max_samples_ == round(0.5 * len(X))
    np.testing.assert_array_equal(a.score_samples(X), b.score_samples(X))


def test_constant_data_and_small_n():
    X = np.ones((10, 3))
    model = IsolationForest(n_estimators=5, random_state=0).fit(X)
    assert model.max_samples_ == 10
    s = model.score_samples(X)
    assert np.allclose(s, s[0])


def test_validation():
    with pytest.raises(ValueError):
        IsolationForest(n_estimators=0)
    with pytest.raises(ValueError):
        IsolationForest(max_samples=1.5)
    with pytest.raises(ValueError):
        IsolationForest(contamination=0.7)
    with pytest.raises(RuntimeError):
        IsolationForest().score_samples(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        IsolationForest().fit(np.array([[np.inf, 0.0], [1.0, 1.0]]))
    model = IsolationForest(n_estimators=3).fit(np.random.default_rng(0).normal(size=(20, 2)))
    with pytest.raises(ValueError):
        model.predict(np.zeros((2, 3)))
