"""Tests for BaggingClassifier and BaggingRegressor."""

from __future__ import annotations

import numpy as np
import pytest

from ensemble_methods_kit import (
    BaggingClassifier,
    BaggingRegressor,
    DecisionTree,
    accuracy_score,
    mean_squared_error,
)
from ensemble_methods_kit.utils import train_test_split


def test_bagging_improves_over_single_tree(binary_cls):
    X, y = binary_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    single = DecisionTree(criterion="gini", max_depth=5, random_state=0).fit(Xtr, ytr)
    single_acc = accuracy_score(yte, single.predict(Xte))
    bag = BaggingClassifier(
        base_estimator=DecisionTree(criterion="gini", max_depth=5),
        n_estimators=15,
        random_state=0,
    ).fit(Xtr, ytr)
    bag_acc = accuracy_score(yte, bag.predict(Xte))
    assert bag_acc >= single_acc - 1e-9


def test_bagging_proba_sums_to_one(binary_cls):
    X, y = binary_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    bag = BaggingClassifier(n_estimators=10, random_state=0).fit(Xtr, ytr)
    proba = bag.predict_proba(Xte)
    assert proba.shape == (len(yte), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_bagging_multiclass(multiclass_cls):
    X, y = multiclass_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    bag = BaggingClassifier(
        base_estimator=DecisionTree(criterion="gini", max_depth=6),
        n_estimators=12,
        random_state=1,
    ).fit(Xtr, ytr)
    assert accuracy_score(yte, bag.predict(Xte)) > 0.85


def test_bagging_reproducible(binary_cls):
    X, y = binary_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    a = BaggingClassifier(n_estimators=10, random_state=7).fit(Xtr, ytr).predict(Xte)
    b = BaggingClassifier(n_estimators=10, random_state=7).fit(Xtr, ytr).predict(Xte)
    assert np.array_equal(a, b)


def test_bagging_no_bootstrap(binary_cls):
    X, y = binary_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    bag = BaggingClassifier(n_estimators=5, bootstrap=False, random_state=0).fit(Xtr, ytr)
    proba = bag.predict_proba(Xte)
    assert proba.shape == (len(yte), 2)


def test_bagging_regressor_predict_shape(regression):
    X, y = regression
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    bag = BaggingRegressor(n_estimators=8, random_state=0).fit(Xtr, ytr)
    pred = bag.predict(Xte)
    assert len(bag.estimators_) == 8
    assert pred.shape == (len(yte),)
    assert pred.dtype == np.float64
    single = bag.predict(Xte[0])
    assert single.shape == (1,)


def test_bagging_regressor_beats_mean(regression):
    X, y = regression
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    bag = BaggingRegressor(
        base_estimator=DecisionTree(criterion="variance", max_depth=6),
        n_estimators=20,
        random_state=0,
    ).fit(Xtr, ytr)
    pred = bag.predict(Xte)
    baseline = np.full(len(yte), float(np.mean(ytr)))
    assert mean_squared_error(yte, pred) < mean_squared_error(yte, baseline)


def test_bagging_regressor_reproducible(regression):
    X, y = regression
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    a = BaggingRegressor(n_estimators=8, random_state=7).fit(Xtr, ytr).predict(Xte)
    b = BaggingRegressor(n_estimators=8, random_state=7).fit(Xtr, ytr).predict(Xte)
    assert np.array_equal(a, b)


def test_bagging_regressor_no_bootstrap(regression):
    X, y = regression
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    bag = BaggingRegressor(n_estimators=5, bootstrap=False, random_state=0).fit(Xtr, ytr)
    assert bag.predict(Xte).shape == (len(yte),)
    with pytest.raises(AttributeError):
        _ = bag.oob_score_


def test_bagging_regressor_oob_score(regression):
    X, y = regression
    bag = BaggingRegressor(n_estimators=25, oob_score=True, random_state=0).fit(X, y)
    assert bag.oob_prediction_.shape == (len(y),)
    assert np.isfinite(bag.oob_prediction_).all()
    assert bag.oob_score_ > 0.0
    with pytest.raises(ValueError):
        BaggingRegressor(n_estimators=5, bootstrap=False, oob_score=True).fit(X, y)


def test_bagging_regressor_max_features_fraction(regression):
    X, y = regression
    bag = BaggingRegressor(n_estimators=3, max_features=0.4, random_state=0).fit(X, y)
    assert bag.estimators_[0].max_features == max(1, int(0.4 * X.shape[1]))
    full = BaggingRegressor(n_estimators=2, random_state=0).fit(X, y)
    assert full.estimators_[0].max_features is None


def test_bagging_regressor_feature_importances(regression):
    X, y = regression
    bag = BaggingRegressor(n_estimators=8, random_state=0).fit(X, y)
    imp = bag.feature_importances_
    assert imp.shape == (X.shape[1],)
    assert np.isclose(imp.sum(), 1.0)


def test_regressor_public_exports():
    from pathlib import Path

    import ensemble_methods_kit as kit

    assert kit.BaggingRegressor is BaggingRegressor
    assert "BaggingRegressor" in kit.__all__
    assert "RandomForestRegressor" in kit.__all__
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")
    cli = (root / "src" / "ensemble_methods_kit" / "__main__.py").read_text(encoding="utf-8")
    for name in ("BaggingRegressor", "RandomForestRegressor"):
        assert name in readme
        assert name in cli
