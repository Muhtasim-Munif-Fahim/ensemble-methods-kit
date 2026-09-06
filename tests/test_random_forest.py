"""Tests for the RandomForestClassifier."""

from __future__ import annotations

import numpy as np

from ensemble_methods_kit import RandomForestClassifier, accuracy_score
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
