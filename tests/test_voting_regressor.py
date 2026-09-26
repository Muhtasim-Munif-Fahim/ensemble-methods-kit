"""Tests for the VotingRegressor."""

from __future__ import annotations

import numpy as np
import pytest

from ensemble_methods_kit import (
    DecisionTree,
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
    mean_squared_error,
    r2_score,
)
from ensemble_methods_kit.utils import train_test_split


def _regression_data(n=160, d=3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, d))
    w = np.array([1.5, -2.0, 0.5][:d], dtype=float)
    y = X @ w + 0.1 * rng.normal(0, 1, n)
    return X, y


def _bases():
    return [
        ("rf", RandomForestRegressor(n_estimators=10, max_depth=4, random_state=1)),
        ("gbr", GradientBoostingRegressor(n_estimators=20, learning_rate=0.1,
                                          max_depth=2, random_state=1)),
        ("dt", DecisionTree(criterion="variance", max_depth=4, random_state=1)),
    ]


class _ConstantRegressor:
    def __init__(self, value):
        self.value = float(value)

    def fit(self, X, y, sample_weight=None):
        self.seen_weight = None if sample_weight is None else np.asarray(sample_weight)
        self._n_features = np.asarray(X).shape[1]
        return self

    def predict(self, X):
        X = np.asarray(X)
        n = 1 if X.ndim == 1 else X.shape[0]
        return np.full(n, self.value, dtype=float)


def test_voting_regressor_averages_predictions():
    X, y = _regression_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    vr = VotingRegressor(_bases()).fit(Xtr, ytr)
    preds = vr.predict(Xte)
    assert preds.shape == (len(yte),)
    assert r2_score(yte, preds) > 0.75
    assert mean_squared_error(yte, preds) < mean_squared_error(yte, np.full_like(yte, ytr.mean()))


def test_voting_regressor_weights_change_the_average():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    vr = VotingRegressor(
        [("a", _ConstantRegressor(0.0)), ("b", _ConstantRegressor(10.0))],
        weights=[1.0, 3.0],
    ).fit(X, y)
    assert np.allclose(vr.predict(X), 7.5)


def test_fit_clones_estimators():
    X, y = _regression_data(n=80)
    tree = DecisionTree(criterion="variance", max_depth=3, random_state=0)
    left = VotingRegressor([("t", tree)]).fit(X, y)
    assert getattr(tree, "tree_", None) is None or tree is not left.estimators_[0]
    assert left.named_estimators_["t"] is not tree


def test_drop_estimator_is_skipped():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    vr = VotingRegressor(
        [("keep", _ConstantRegressor(5.0)), ("gone", "drop")],
        weights=[1.0, 50.0],
    ).fit(X, y)
    assert vr.named_estimators_["gone"] == "drop"
    assert len(vr.estimators_) == 1
    assert np.allclose(vr.predict(X), 5.0)


def test_sample_weight_forwarded():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0.0, 1.0, 0.0, 1.0])
    weights = np.array([1.0, 1.0, 2.0, 2.0])
    stub = _ConstantRegressor(0.0)
    vr = VotingRegressor([("s", stub)]).fit(X, y, sample_weight=weights)
    assert np.allclose(vr.named_estimators_["s"].seen_weight, weights)


def test_transform_and_feature_names():
    X, y = _regression_data(n=80)
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.3, random_state=42)
    vr = VotingRegressor(_bases()).fit(Xtr, ytr)
    transformed = vr.transform(Xtr[:4])
    assert transformed.shape == (4, 3)
    assert vr.get_feature_names_out().tolist() == [
        "votingregressor_rf",
        "votingregressor_gbr",
        "votingregressor_dt",
    ]


def test_feature_count_mismatch_and_not_fitted():
    X, y = _regression_data(n=60)
    vr = VotingRegressor(
        [("t", DecisionTree(criterion="variance", max_depth=2, random_state=0))]
    ).fit(X, y)
    with pytest.raises(ValueError, match="features"):
        vr.predict(X[:, :1])
    with pytest.raises(RuntimeError, match="not fitted"):
        VotingRegressor(_bases()).predict(X)
    with pytest.raises(ValueError, match="weights"):
        VotingRegressor(_bases(), weights=[1.0, 2.0])
    with pytest.raises(ValueError, match="unique"):
        VotingRegressor([("a", _ConstantRegressor(0)), ("a", _ConstantRegressor(1))])
