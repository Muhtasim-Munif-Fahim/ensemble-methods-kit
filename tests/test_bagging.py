"""Tests for the BaggingClassifier."""

from __future__ import annotations

import numpy as np

from ensemble_methods_kit import BaggingClassifier, DecisionTree, accuracy_score
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
