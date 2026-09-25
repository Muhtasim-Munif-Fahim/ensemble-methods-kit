"""Tests for RandomForestClassifier and RandomForestRegressor."""

from __future__ import annotations

import numpy as np
import pytest

from ensemble_methods_kit import (
    RandomForestClassifier,
    RandomForestRegressor,
    accuracy_score,
    mean_squared_error,
)
from ensemble_methods_kit.utils import train_test_split


def test_random_forest_multiclass(multiclass_cls):
    X, y = multiclass_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=20, max_depth=6, random_state=0).fit(Xtr, ytr)
    assert accuracy_score(yte, rf.predict(Xte)) > 0.9


def test_random_forest_proba_sums_to_one(binary_cls):
    X, y = binary_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(Xtr, ytr)
    proba = rf.predict_proba(Xte)
    assert proba.shape == (len(yte), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_random_forest_beats_single_tree(binary_cls):
    from ensemble_methods_kit import DecisionTree

    X, y = binary_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    single = DecisionTree(criterion="gini", max_depth=6, random_state=0).fit(Xtr, ytr)
    rf = RandomForestClassifier(n_estimators=30, max_depth=6, random_state=0).fit(Xtr, ytr)
    assert accuracy_score(yte, rf.predict(Xte)) >= accuracy_score(yte, single.predict(Xte)) - 1e-9


def test_random_forest_reproducible(binary_cls):
    X, y = binary_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    a = RandomForestClassifier(n_estimators=10, random_state=3).fit(Xtr, ytr).predict(Xte)
    b = RandomForestClassifier(n_estimators=10, random_state=3).fit(Xtr, ytr).predict(Xte)
    assert np.array_equal(a, b)


def test_random_forest_regressor_predict_shape(regression):
    X, y = regression
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    rf = RandomForestRegressor(n_estimators=10, max_depth=5, random_state=0).fit(Xtr, ytr)
    pred = rf.predict(Xte)
    assert len(rf.estimators_) == 10
    assert pred.shape == (len(yte),)
    assert rf.predict(Xte[0]).shape == (1,)


def test_random_forest_regressor_beats_mean(regression):
    X, y = regression
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    rf = RandomForestRegressor(
        n_estimators=30, max_depth=6, max_features="sqrt", random_state=0
    ).fit(Xtr, ytr)
    pred = rf.predict(Xte)
    baseline = np.full(len(yte), float(np.mean(ytr)))
    assert mean_squared_error(yte, pred) < mean_squared_error(yte, baseline)


def test_random_forest_regressor_max_features(regression):
    X, y = regression
    rf = RandomForestRegressor(n_estimators=4, max_features=2, random_state=0).fit(X, y)
    assert all(tree.max_features == 2 for tree in rf.estimators_)
    assert rf.predict(X).shape == (len(y),)
    rf_sqrt = RandomForestRegressor(n_estimators=4, max_features="sqrt", random_state=0).fit(X, y)
    assert all(tree.max_features == "sqrt" for tree in rf_sqrt.estimators_)
    rf_log = RandomForestRegressor(n_estimators=3, max_features="log2", random_state=1).fit(X, y)
    assert all(tree.max_features == "log2" for tree in rf_log.estimators_)
    imp = rf.feature_importances_
    assert imp.shape == (X.shape[1],)
    assert np.isclose(imp.sum(), 1.0)


def test_random_forest_regressor_reproducible(regression):
    X, y = regression
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    a = RandomForestRegressor(n_estimators=8, random_state=3).fit(Xtr, ytr).predict(Xte)
    b = RandomForestRegressor(n_estimators=8, random_state=3).fit(Xtr, ytr).predict(Xte)
    assert np.array_equal(a, b)


def test_random_forest_regressor_oob(regression):
    X, y = regression
    rf = RandomForestRegressor(n_estimators=20, oob_score=True, random_state=0).fit(X, y)
    assert rf.oob_prediction_.shape == (len(y),)
    assert np.isfinite(rf.oob_score_)
    assert rf.oob_score_ > 0.0
    plain = RandomForestRegressor(n_estimators=4, random_state=0).fit(X, y)
    with pytest.raises(AttributeError):
        _ = plain.oob_score_
