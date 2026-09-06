"""Tests for stacking and blending meta-ensembles and the logistic regression."""

from __future__ import annotations

import numpy as np

from ensemble_methods_kit import (
    BlendingClassifier,
    DecisionTree,
    GradientBoostingClassifier,
    LogisticRegression,
    RandomForestClassifier,
    StackingClassifier,
    accuracy_score,
    r2_score,
)
from ensemble_methods_kit.utils import train_test_split


def _bases():
    return [
        ("rf", RandomForestClassifier(n_estimators=10, max_depth=5, random_state=1)),
        ("gbc", GradientBoostingClassifier(n_estimators=15, learning_rate=0.1,
                                           max_depth=2, random_state=1)),
        ("dt", DecisionTree(criterion="gini", max_depth=5, random_state=1)),
    ]


def test_logistic_regression_multiclass(binary_cls):
    from ensemble_methods_kit import train_test_split

    X, y = binary_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    lr = LogisticRegression(lr=0.5, n_iter=300, l2=1.0, random_state=0).fit(Xtr, ytr)
    assert lr.classes_.tolist() == [0, 1]
    assert accuracy_score(yte, lr.predict(Xte)) > 0.85
    proba = lr.predict_proba(Xte)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_stacking_beats_weak_base(binary_cls):
    X, y = binary_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    weak = DecisionTree(criterion="gini", max_depth=1, random_state=0)
    stack = StackingClassifier([("weak", weak)], cv=3, random_state=0).fit(Xtr, ytr)
    base_acc = accuracy_score(yte, weak.fit(Xtr, ytr).predict(Xte))
    stack_acc = accuracy_score(yte, stack.predict(Xte))
    assert stack_acc >= base_acc


def test_stacking_proba_sums_to_one(multiclass_cls):
    X, y = multiclass_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    stack = StackingClassifier(_bases(), cv=3, random_state=0).fit(Xtr, ytr)
    proba = stack.predict_proba(Xte)
    assert proba.shape == (len(yte), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert accuracy_score(yte, stack.predict(Xte)) > 0.9


def test_stacking_passthrough(binary_cls):
    X, y = binary_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    stack = StackingClassifier(_bases(), cv=3, passthrough=True, random_state=0).fit(Xtr, ytr)
    assert stack.predict_proba(Xte).shape == (len(yte), 2)


def test_blending_runs_and_predicts(binary_cls):
    X, y = binary_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    blend = BlendingClassifier(_bases(), validation_fraction=0.3, random_state=0).fit(Xtr, ytr)
    proba = blend.predict_proba(Xte)
    assert proba.shape == (len(yte), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    preds = blend.predict(Xte)
    assert set(np.unique(preds)).issubset({0, 1})


def test_blending_multiclass(multiclass_cls):
    X, y = multiclass_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    blend = BlendingClassifier(_bases(), validation_fraction=0.3, random_state=0).fit(Xtr, ytr)
    assert blend.predict_proba(Xte).shape == (len(yte), 3)
    assert accuracy_score(yte, blend.predict(Xte)) > 0.85


def test_stacking_reproducible(binary_cls):
    X, y = binary_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    a = StackingClassifier(_bases(), cv=3, random_state=0).fit(Xtr, ytr).predict(Xte)
    b = StackingClassifier(_bases(), cv=3, random_state=0).fit(Xtr, ytr).predict(Xte)
    assert np.array_equal(a, b)
