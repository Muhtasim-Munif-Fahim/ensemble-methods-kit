"""Tests for Mean Decrease Impurity (MDI) feature importances."""

from __future__ import annotations

import numpy as np
import pytest

from ensemble_methods_kit import (
    DecisionTree,
    ExtraTreesClassifier,
    RandomForestClassifier,
    mean_decrease_impurity,
)


def _signal_and_noise(n: int = 240, random_state: int = 0):
    """Binary labels that depend only on the first feature."""
    rng = np.random.default_rng(random_state)
    signal = rng.normal(0, 1, n)
    X = np.column_stack(
        [signal, rng.normal(0, 1, n), rng.normal(0, 1, n), np.zeros(n)]
    )
    y = (signal > 0).astype(int)
    return X, y


def test_decision_tree_feature_importances_shape_and_sum():
    X, y = _signal_and_noise()
    tree = DecisionTree(criterion="gini", max_depth=4, random_state=0).fit(X, y)
    imp = tree.feature_importances_
    assert imp.shape == (4,)
    assert imp.dtype == float
    assert imp.min() >= 0.0
    assert abs(imp.sum() - 1.0) < 1e-9


def test_decision_tree_ranks_informative_feature():
    X, y = _signal_and_noise()
    tree = DecisionTree(criterion="gini", max_depth=4, random_state=0).fit(X, y)
    imp = tree.feature_importances_
    assert imp.argmax() == 0
    assert imp[0] > imp[1]
    # constant column can never be chosen
    assert imp[3] == 0.0


def test_decision_tree_importances_before_fit_raises():
    tree = DecisionTree()
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = tree.feature_importances_


def test_mean_decrease_impurity_matches_tree_property():
    X, y = _signal_and_noise()
    tree = DecisionTree(criterion="entropy", max_depth=3, random_state=1).fit(X, y)
    np.testing.assert_allclose(mean_decrease_impurity(tree), tree.feature_importances_)


def test_mean_decrease_impurity_empty_raises():
    with pytest.raises(RuntimeError, match="not fitted"):
        mean_decrease_impurity([])


def test_random_forest_feature_importances_shape_and_sum():
    X, y = _signal_and_noise()
    rf = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=0).fit(X, y)
    imp = rf.feature_importances_
    assert imp.shape == (4,)
    assert abs(imp.sum() - 1.0) < 1e-9
    assert imp.min() >= 0.0


def test_random_forest_ranks_informative_feature():
    X, y = _signal_and_noise()
    rf = RandomForestClassifier(n_estimators=25, max_depth=5, random_state=0).fit(X, y)
    assert rf.feature_importances_.argmax() == 0
    assert rf.feature_importances_[3] == 0.0


def test_random_forest_importances_average_trees():
    X, y = _signal_and_noise()
    rf = RandomForestClassifier(n_estimators=12, max_depth=4, random_state=2).fit(X, y)
    expected = mean_decrease_impurity(rf.estimators_)
    np.testing.assert_allclose(rf.feature_importances_, expected)


def test_random_forest_importances_before_fit_raises():
    rf = RandomForestClassifier(n_estimators=5)
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = rf.feature_importances_


def test_extra_trees_feature_importances_shape_and_sum():
    X, y = _signal_and_noise()
    et = ExtraTreesClassifier(n_estimators=20, max_depth=5, random_state=0).fit(X, y)
    imp = et.feature_importances_
    assert imp.shape == (4,)
    assert abs(imp.sum() - 1.0) < 1e-9
    assert imp.min() >= 0.0


def test_extra_trees_ranks_informative_feature():
    X, y = _signal_and_noise()
    et = ExtraTreesClassifier(n_estimators=30, max_depth=5, random_state=0).fit(X, y)
    assert et.feature_importances_.argmax() == 0
    assert et.feature_importances_[3] == 0.0


def test_extra_trees_importances_average_trees():
    X, y = _signal_and_noise()
    et = ExtraTreesClassifier(n_estimators=12, max_depth=4, random_state=3).fit(X, y)
    expected = mean_decrease_impurity(et.estimators_)
    np.testing.assert_allclose(et.feature_importances_, expected)


def test_extra_trees_importances_before_fit_raises():
    et = ExtraTreesClassifier(n_estimators=5)
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = et.feature_importances_


def test_stump_on_one_feature_is_unit_mass():
    X = np.array([[0.0], [1.0], [0.1], [0.9]])
    y = np.array([0, 1, 0, 1])
    tree = DecisionTree(criterion="gini", max_depth=1, random_state=0).fit(X, y)
    np.testing.assert_allclose(tree.feature_importances_, [1.0])
