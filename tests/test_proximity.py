"""Tests for forest proximities and proximity outlier scores."""

from __future__ import annotations

import numpy as np
import pytest

from ensemble_methods_kit import (
    DecisionTree,
    ExtraTreesClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
    forest_apply,
    proximity_matrix,
    proximity_outlier_scores,
)


@pytest.fixture
def fitted(binary_cls):
    X, y = binary_cls
    rf = RandomForestClassifier(n_estimators=30, max_depth=6, random_state=0).fit(X, y)
    return rf, X, y


def test_apply_shape_and_matches_trees(fitted):
    rf, X, _ = fitted
    leaves = forest_apply(rf, X)
    assert leaves.shape == (X.shape[0], 30)
    np.testing.assert_array_equal(leaves[:, 3], rf.estimators_[3].apply(X))


def test_proximity_properties(fitted):
    rf, X, y = fitted
    P = proximity_matrix(rf, X)
    assert P.shape == (len(X), len(X))
    np.testing.assert_allclose(P, P.T)
    np.testing.assert_allclose(np.diag(P), 1.0)
    assert P.min() >= 0.0 and P.max() <= 1.0
    # values are multiples of 1/T
    np.testing.assert_allclose(P * 30, np.round(P * 30))
    same = P[y[:, None] == y[None, :]].mean()
    diff = P[y[:, None] != y[None, :]].mean()
    assert same > 5 * diff


def test_proximity_cross_block_matches_full(fitted):
    rf, X, _ = fitted
    full = proximity_matrix(rf, X)
    block = proximity_matrix(rf, X[:10], X[50:70])
    np.testing.assert_allclose(block, full[:10, 50:70])


def test_brute_force_single_tree():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 2))
    y = (X[:, 0] > 0).astype(int)
    rf = RandomForestClassifier(n_estimators=1, max_depth=3, random_state=1).fit(X, y)
    leaf = rf.estimators_[0].apply(X)
    expected = (leaf[:, None] == leaf[None, :]).astype(float)
    np.testing.assert_array_equal(proximity_matrix(rf, X), expected)


def test_outlier_score_flags_mislabeled_row(binary_cls):
    X, y = binary_cls
    X = np.vstack([X, [4.0, 4.0, 4.0]])  # deep inside class-1 territory...
    y = np.append(y, 0)  # ...but labelled class 0
    rf = ExtraTreesClassifier(n_estimators=60, max_depth=None, random_state=0).fit(X, y)
    scores = proximity_outlier_scores(rf, X, y)
    assert scores.shape == (len(X),)
    assert np.argmax(scores) == len(X) - 1
    assert scores[-1] > 10


def test_works_for_regressor():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(60, 2))
    y = X[:, 0] * 3
    rf = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)
    assert proximity_matrix(rf, X).shape == (60, 60)


def test_errors(binary_cls, fitted):
    X, y = binary_cls
    rf, _, _ = fitted
    with pytest.raises(RuntimeError):
        proximity_matrix(RandomForestClassifier(n_estimators=3), X)
    class NotTrees:
        estimators_ = [object()]
    with pytest.raises(TypeError):
        forest_apply(NotTrees(), X)
    with pytest.raises(ValueError):
        proximity_outlier_scores(rf, X, y[:-1])


def test_singleton_class_scores_zero(binary_cls):
    X, y = binary_cls
    y = y.copy()
    y[0] = 7
    rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
    assert proximity_outlier_scores(rf, X, y)[0] == 0.0
