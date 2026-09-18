"""Tests for the ExtraTreesClassifier."""

from __future__ import annotations

import numpy as np

from ensemble_methods_kit import DecisionTree, ExtraTreesClassifier, accuracy_score
from ensemble_methods_kit.utils import train_test_split


def test_extra_trees_trains_and_predicts(binary_cls):
    X, y = binary_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    et = ExtraTreesClassifier(n_estimators=15, max_depth=6, random_state=0).fit(Xtr, ytr)
    preds = et.predict(Xte)
    assert preds.shape == yte.shape
    assert set(np.unique(preds)).issubset({0, 1})
    assert accuracy_score(yte, preds) > 0.9
    assert len(et.estimators_) == 15
    assert all(tree.splitter == "random" for tree in et.estimators_)


def test_extra_trees_multiclass(multiclass_cls):
    X, y = multiclass_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    et = ExtraTreesClassifier(n_estimators=20, max_depth=6, random_state=0).fit(Xtr, ytr)
    assert accuracy_score(yte, et.predict(Xte)) > 0.9


def test_extra_trees_proba_sums_to_one(binary_cls):
    X, y = binary_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    et = ExtraTreesClassifier(n_estimators=10, random_state=0).fit(Xtr, ytr)
    proba = et.predict_proba(Xte)
    assert proba.shape == (len(yte), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_extra_trees_beats_single_tree(binary_cls):
    X, y = binary_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    single = DecisionTree(criterion="gini", max_depth=6, random_state=0).fit(Xtr, ytr)
    et = ExtraTreesClassifier(n_estimators=30, max_depth=6, random_state=0).fit(Xtr, ytr)
    assert accuracy_score(yte, et.predict(Xte)) >= accuracy_score(yte, single.predict(Xte)) - 1e-9


def test_extra_trees_reproducible(binary_cls):
    X, y = binary_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    a = ExtraTreesClassifier(n_estimators=10, random_state=3).fit(Xtr, ytr).predict(Xte)
    b = ExtraTreesClassifier(n_estimators=10, random_state=3).fit(Xtr, ytr).predict(Xte)
    assert np.array_equal(a, b)


def test_extra_trees_default_no_bootstrap():
    et = ExtraTreesClassifier(n_estimators=3)
    assert et.bootstrap is False


def test_decision_tree_invalid_splitter():
    import pytest

    with pytest.raises(ValueError, match="splitter"):
        DecisionTree(splitter="fancy")


def test_decision_tree_random_splitter(binary_cls):
    X, y = binary_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    tree = DecisionTree(
        criterion="gini", max_depth=5, splitter="random", random_state=0
    ).fit(Xtr, ytr)
    preds = tree.predict(Xte)
    assert preds.shape == yte.shape
    assert accuracy_score(yte, preds) > 0.85
    proba = tree.predict_proba(Xte)
    assert np.allclose(proba.sum(axis=1), 1.0)
