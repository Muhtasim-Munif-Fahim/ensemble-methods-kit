"""Tests for RotationForestClassifier."""

from __future__ import annotations

import numpy as np
import pytest

from ensemble_methods_kit import (
    DecisionTree,
    RotationForestClassifier,
    accuracy_score,
    train_test_split,
)


def test_rotation_forest_fit_predict_shape(binary_cls):
    X, y = binary_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = RotationForestClassifier(n_estimators=8, n_features_per_subset=2, random_state=0).fit(Xtr, ytr)
    preds = clf.predict(Xte)
    proba = clf.predict_proba(Xte)
    assert preds.shape == yte.shape
    assert proba.shape == (len(yte), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert len(clf.estimators_) == 8
    assert len(clf.rotation_matrices_) == 8
    assert clf._n_features_in_ == X.shape[1]
    for R in clf.rotation_matrices_:
        assert R.shape == (X.shape[1], X.shape[1])
        assert np.all(np.isfinite(R))


def test_rotation_forest_beats_chance_binary(binary_cls):
    X, y = binary_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = RotationForestClassifier(n_estimators=15, n_features_per_subset=2, random_state=0).fit(Xtr, ytr)
    assert accuracy_score(yte, clf.predict(Xte)) >= 0.85


def test_rotation_forest_multiclass(multiclass_cls):
    X, y = multiclass_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf = RotationForestClassifier(n_estimators=12, n_features_per_subset=2, random_state=1).fit(Xtr, ytr)
    preds = clf.predict(Xte)
    proba = clf.predict_proba(Xte)
    assert preds.shape == yte.shape
    assert proba.shape == (len(yte), 3)
    assert accuracy_score(yte, preds) >= 0.8


def test_rotation_forest_reproducible(binary_cls):
    X, y = binary_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    a = RotationForestClassifier(n_estimators=6, n_features_per_subset=2, random_state=7).fit(Xtr, ytr).predict_proba(Xte)
    b = RotationForestClassifier(n_estimators=6, n_features_per_subset=2, random_state=7).fit(Xtr, ytr).predict_proba(Xte)
    assert np.allclose(a, b)


def test_rotation_forest_n_subsets(binary_cls):
    X, y = binary_cls
    clf = RotationForestClassifier(n_estimators=5, n_subsets=2, random_state=0).fit(X, y)
    assert len(clf.estimators_) == 5
    assert clf.predict(X[:5]).shape == (5,)


def test_rotation_forest_feature_importances(binary_cls):
    X, y = binary_cls
    clf = RotationForestClassifier(n_estimators=8, n_features_per_subset=2, random_state=0).fit(X, y)
    imp = clf.feature_importances_
    assert imp.shape == (X.shape[1],)
    assert np.all(imp >= 0)
    assert abs(imp.sum() - 1.0) < 1e-6


def test_rotation_forest_custom_tree(binary_cls):
    X, y = binary_cls
    base = DecisionTree(criterion="entropy", max_depth=3)
    clf = RotationForestClassifier(
        base_estimator=base, n_estimators=5, n_features_per_subset=2, random_state=0
    ).fit(X, y)
    assert all(t.criterion == "entropy" for t in clf.estimators_)
    assert all(t.max_depth == 3 for t in clf.estimators_)


def test_rotation_forest_rejects_bad_params():
    with pytest.raises(ValueError):
        RotationForestClassifier(n_estimators=0)
    with pytest.raises(ValueError):
        RotationForestClassifier(n_features_per_subset=0)
    with pytest.raises(ValueError):
        RotationForestClassifier(max_samples=0.0)
    with pytest.raises(ValueError):
        RotationForestClassifier(n_subsets=0)


def test_rotation_forest_predict_before_fit_raises(binary_cls):
    X, y = binary_cls
    with pytest.raises(RuntimeError):
        RotationForestClassifier().predict(X)


def test_rotation_forest_export():
    from ensemble_methods_kit import RotationForestClassifier as RFC
    assert callable(RFC)
