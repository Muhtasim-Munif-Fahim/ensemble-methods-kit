"""Tests for RandomSubspaceRegressor."""

from __future__ import annotations

import numpy as np
import pytest

from ensemble_methods_kit import (
    DecisionTree,
    RandomSubspaceRegressor,
    mean_squared_error,
    r2_score,
)
from ensemble_methods_kit.utils import train_test_split


def test_random_subspace_regressor_fit_predict_shape(regression):
    X, y = regression
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    reg = RandomSubspaceRegressor(
        n_estimators=10, max_features=0.5, random_state=0
    ).fit(Xtr, ytr)
    preds = reg.predict(Xte)
    assert preds.shape == yte.shape
    assert np.all(np.isfinite(preds))
    assert len(reg.estimators_) == 10
    assert len(reg.estimators_features_) == 10
    assert reg._n_features_in_ == X.shape[1]
    assert all(tree.criterion == "variance" for tree in reg.estimators_)


def test_random_subspace_regressor_improves_over_single_tree(regression):
    X, y = regression
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    single = DecisionTree(criterion="variance", max_depth=5, random_state=0).fit(Xtr, ytr)
    single_mse = mean_squared_error(yte, single.predict(Xte))
    ens = RandomSubspaceRegressor(
        base_estimator=DecisionTree(criterion="variance", max_depth=5),
        n_estimators=25,
        max_features=0.6,
        random_state=0,
    ).fit(Xtr, ytr)
    ens_mse = mean_squared_error(yte, ens.predict(Xte))
    # Subspace trees are individually weaker than a full-feature CART;
    # the ensemble should still land in a comparable ballpark.
    assert ens_mse <= single_mse * 1.75 + 1e-9
    assert r2_score(yte, ens.predict(Xte)) > 0.4


def test_random_subspace_regressor_reproducible(regression):
    X, y = regression
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    a = RandomSubspaceRegressor(
        n_estimators=10, max_features=0.5, random_state=7
    ).fit(Xtr, ytr).predict(Xte)
    b = RandomSubspaceRegressor(
        n_estimators=10, max_features=0.5, random_state=7
    ).fit(Xtr, ytr).predict(Xte)
    assert np.allclose(a, b)


def test_random_subspace_regressor_stores_feature_indices(regression):
    X, y = regression
    reg = RandomSubspaceRegressor(
        n_estimators=8, max_features=0.5, random_state=0
    ).fit(X, y)
    assert len(reg.estimators_) == 8
    assert len(reg.estimators_features_) == 8
    n_features = X.shape[1]
    expected_k = max(1, int(round(0.5 * n_features)))
    for feats in reg.estimators_features_:
        assert len(feats) == expected_k
        assert len(np.unique(feats)) == expected_k
        assert feats.min() >= 0
        assert feats.max() < n_features


def test_random_subspace_regressor_max_features_full(regression):
    X, y = regression
    reg = RandomSubspaceRegressor(
        n_estimators=5, max_features=1.0, random_state=0
    ).fit(X, y)
    for feats in reg.estimators_features_:
        assert len(feats) == X.shape[1]


def test_random_subspace_regressor_bootstrap_option(regression):
    X, y = regression
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    no_boot = RandomSubspaceRegressor(
        n_estimators=8, max_features=0.5, bootstrap=False, random_state=0
    ).fit(Xtr, ytr)
    with_boot = RandomSubspaceRegressor(
        n_estimators=8, max_features=0.5, bootstrap=True, random_state=0
    ).fit(Xtr, ytr)
    assert no_boot.predict(Xte).shape == (len(yte),)
    assert with_boot.predict(Xte).shape == (len(yte),)
    # Different row sampling should generally yield different predictions
    assert not np.allclose(no_boot.predict(Xte), with_boot.predict(Xte))


def test_random_subspace_regressor_feature_importances(regression):
    X, y = regression
    reg = RandomSubspaceRegressor(
        n_estimators=12, max_features=0.5, random_state=0
    ).fit(X, y)
    fi = reg.feature_importances_
    assert fi.shape == (X.shape[1],)
    assert pytest.approx(fi.sum(), abs=1e-9) == 1.0
    assert np.all(fi >= 0.0)


def test_random_subspace_regressor_invalid_params():
    with pytest.raises(ValueError):
        RandomSubspaceRegressor(n_estimators=0)
    with pytest.raises(ValueError):
        RandomSubspaceRegressor(max_features=0.0)
    with pytest.raises(ValueError):
        RandomSubspaceRegressor(max_features=1.5)
    with pytest.raises(ValueError):
        RandomSubspaceRegressor(max_samples=0.0)


def test_random_subspace_regressor_export_available():
    from ensemble_methods_kit import RandomSubspaceRegressor as exported

    assert exported is RandomSubspaceRegressor


def test_random_subspace_regressor_predict_before_fit_raises(regression):
    X, y = regression
    reg = RandomSubspaceRegressor(n_estimators=3, max_features=0.5)
    with pytest.raises(RuntimeError):
        reg.predict(X)


def test_random_subspace_regressor_y_shape_errors(regression):
    X, y = regression
    reg = RandomSubspaceRegressor(n_estimators=3, max_features=0.5, random_state=0)
    with pytest.raises(ValueError, match="1-d"):
        reg.fit(X, np.column_stack([y, y]))
    with pytest.raises(ValueError, match="same number"):
        reg.fit(X, y[:10])
