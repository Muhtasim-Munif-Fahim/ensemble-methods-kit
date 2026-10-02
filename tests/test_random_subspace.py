"""Tests for RandomSubspaceClassifier."""

from __future__ import annotations

import numpy as np
import pytest

from ensemble_methods_kit import (
    DecisionTree,
    RandomSubspaceClassifier,
    accuracy_score,
)
from ensemble_methods_kit.utils import train_test_split


def test_random_subspace_proba_sums_to_one(binary_cls):
    X, y = binary_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = RandomSubspaceClassifier(n_estimators=10, max_features=0.5, random_state=0).fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)
    assert proba.shape == (len(yte), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_random_subspace_improves_over_single_tree(binary_cls):
    X, y = binary_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    single = DecisionTree(criterion="gini", max_depth=5, random_state=0).fit(Xtr, ytr)
    single_acc = accuracy_score(yte, single.predict(Xte))
    ens = RandomSubspaceClassifier(
        base_estimator=DecisionTree(criterion="gini", max_depth=5),
        n_estimators=20,
        max_features=0.5,
        random_state=0,
    ).fit(Xtr, ytr)
    ens_acc = accuracy_score(yte, ens.predict(Xte))
    assert ens_acc >= single_acc - 0.05


def test_random_subspace_multiclass(multiclass_cls):
    X, y = multiclass_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf = RandomSubspaceClassifier(
        base_estimator=DecisionTree(criterion="gini", max_depth=6),
        n_estimators=15,
        max_features=0.6,
        random_state=1,
    ).fit(Xtr, ytr)
    assert accuracy_score(yte, clf.predict(Xte)) > 0.8


def test_random_subspace_reproducible(binary_cls):
    X, y = binary_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    a = RandomSubspaceClassifier(n_estimators=10, max_features=0.5, random_state=7).fit(Xtr, ytr).predict(Xte)
    b = RandomSubspaceClassifier(n_estimators=10, max_features=0.5, random_state=7).fit(Xtr, ytr).predict(Xte)
    assert np.array_equal(a, b)


def test_random_subspace_stores_feature_indices(binary_cls):
    X, y = binary_cls
    clf = RandomSubspaceClassifier(n_estimators=8, max_features=0.5, random_state=0).fit(X, y)
    assert len(clf.estimators_) == 8
    assert len(clf.estimators_features_) == 8
    n_features = X.shape[1]
    expected_k = max(1, int(round(0.5 * n_features)))
    for feats in clf.estimators_features_:
        assert len(feats) == expected_k
        assert len(np.unique(feats)) == expected_k
        assert feats.min() >= 0
        assert feats.max() < n_features


def test_random_subspace_bootstrap_option(binary_cls):
    X, y = binary_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = RandomSubspaceClassifier(
        n_estimators=8, max_features=0.5, bootstrap=True, random_state=0
    ).fit(Xtr, ytr)
    assert clf.predict(Xte).shape == (len(yte),)


def test_random_subspace_feature_importances(binary_cls):
    X, y = binary_cls
    clf = RandomSubspaceClassifier(n_estimators=12, max_features=0.5, random_state=0).fit(X, y)
    fi = clf.feature_importances_
    assert fi.shape == (X.shape[1],)
    assert pytest.approx(fi.sum(), abs=1e-9) == 1.0
    assert np.all(fi >= 0.0)


def test_random_subspace_invalid_params():
    with pytest.raises(ValueError):
        RandomSubspaceClassifier(n_estimators=0)
    with pytest.raises(ValueError):
        RandomSubspaceClassifier(max_features=0.0)
    with pytest.raises(ValueError):
        RandomSubspaceClassifier(max_features=1.5)


def test_random_subspace_export_available():
    from ensemble_methods_kit import RandomSubspaceClassifier as exported

    assert exported is RandomSubspaceClassifier


def test_random_subspace_predict_before_fit_raises(binary_cls):
    X, y = binary_cls
    clf = RandomSubspaceClassifier(n_estimators=3, max_features=0.5)
    with pytest.raises(RuntimeError):
        clf.predict(X)
