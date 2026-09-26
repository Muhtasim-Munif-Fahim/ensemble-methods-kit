"""Tests for the ExtraTreesRegressor."""

from __future__ import annotations

import numpy as np

from ensemble_methods_kit import DecisionTree, ExtraTreesRegressor, mean_squared_error, r2_score
from ensemble_methods_kit.utils import train_test_split


def test_extra_trees_regressor_trains_and_predicts(regression):
    X, y = regression
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    et = ExtraTreesRegressor(n_estimators=20, max_depth=6, random_state=0).fit(Xtr, ytr)
    preds = et.predict(Xte)
    assert preds.shape == yte.shape
    assert np.all(np.isfinite(preds))
    assert r2_score(yte, preds) > 0.7
    assert len(et.estimators_) == 20
    assert all(tree.splitter == "random" for tree in et.estimators_)
    assert all(tree.criterion == "variance" for tree in et.estimators_)


def test_extra_trees_regressor_competitive_with_single_tree(regression):
    X, y = regression
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    single = DecisionTree(criterion="variance", max_depth=6, random_state=0).fit(Xtr, ytr)
    et = ExtraTreesRegressor(n_estimators=60, max_depth=6, random_state=0).fit(Xtr, ytr)
    # Random cuts are individually weaker than greedy ones; the ensemble
    # should still land in the same ballpark as a single CART tree.
    et_mse = mean_squared_error(yte, et.predict(Xte))
    tree_mse = mean_squared_error(yte, single.predict(Xte))
    assert et_mse <= tree_mse * 1.5 + 1e-9
    assert r2_score(yte, et.predict(Xte)) > 0.6


def test_extra_trees_regressor_reproducible(regression):
    X, y = regression
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    a = ExtraTreesRegressor(n_estimators=15, random_state=3).fit(Xtr, ytr).predict(Xte)
    b = ExtraTreesRegressor(n_estimators=15, random_state=3).fit(Xtr, ytr).predict(Xte)
    assert np.allclose(a, b)


def test_extra_trees_regressor_default_no_bootstrap():
    et = ExtraTreesRegressor(n_estimators=3)
    assert et.bootstrap is False


def test_extra_trees_regressor_feature_importances(regression):
    X, y = regression
    et = ExtraTreesRegressor(n_estimators=25, max_depth=5, random_state=0).fit(X, y)
    assert et.feature_importances_.shape == (X.shape[1],)
    assert np.isclose(et.feature_importances_.sum(), 1.0)
    assert (et.feature_importances_ >= 0).all()
