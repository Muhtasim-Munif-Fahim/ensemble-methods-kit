"""Tests for metrics, train/test splitting and the CART tree."""

from __future__ import annotations

import numpy as np
import pytest

from ensemble_methods_kit.utils import (
    DecisionTree,
    accuracy_score,
    clone_estimator,
    log_loss,
    mean_squared_error,
    r2_score,
    train_test_split,
)


def test_train_test_split_shapes(binary_cls):
    X, y = binary_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    assert Xtr.shape[0] + Xte.shape[0] == len(y)
    assert Xte.shape[1] == X.shape[1]
    assert ytr.shape[0] == Xtr.shape[0]
    assert yte.shape[0] == Xte.shape[0]


def test_train_test_split_stratify_preserves_proportions(binary_cls):
    X, y = binary_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    full_pos = y.mean()
    assert abs(ytr.mean() - full_pos) < 0.05
    assert abs(yte.mean() - full_pos) < 0.05


def test_train_test_split_no_stratify(binary_cls):
    X, y = binary_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1)
    assert Xte.shape[0] == int(round(0.3 * len(y)))


def test_accuracy_score():
    assert accuracy_score([0, 1, 1, 0], [0, 1, 0, 0]) == 0.75
    assert accuracy_score([1, 1, 1], [1, 1, 1]) == 1.0
    assert accuracy_score([0, 1], [1, 0]) == 0.0


def test_mean_squared_error_and_r2():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])
    assert mean_squared_error(y_true, y_pred) == pytest.approx(1 / 3)
    # r2 of a perfect mean predictor on a constant target is 1
    assert r2_score(np.full(5, 3.0), np.full(5, 3.0)) == 1.0
    assert r2_score(y_true, y_pred) < 1.0


def test_log_loss_bounds(binary_cls):
    y_true = np.array([0, 1])
    proba = np.array([[0.9, 0.1], [0.1, 0.9]])
    loss = log_loss(y_true, proba)
    assert 0.0 < loss < 1.0
    # confident wrong prediction is penalised heavily
    wrong = np.array([[0.01, 0.99], [0.01, 0.99]])
    assert log_loss(np.array([1, 0]), wrong) > loss


def test_decision_tree_classifier_perfect_separable(binary_cls):
    X, y = binary_cls
    from ensemble_methods_kit.utils import train_test_split

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    tree = DecisionTree(criterion="gini", max_depth=5, random_state=0).fit(Xtr, ytr)
    assert accuracy_score(yte, tree.predict(Xte)) > 0.95
    proba = tree.predict_proba(Xte)
    assert proba.shape == (len(yte), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_decision_tree_multiclass(multiclass_cls):
    X, y = multiclass_cls
    from ensemble_methods_kit.utils import train_test_split

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    tree = DecisionTree(criterion="entropy", max_depth=6, random_state=1).fit(Xtr, ytr)
    assert tree.classes_.tolist() == [0, 1, 2]
    assert accuracy_score(yte, tree.predict(Xte)) > 0.9
    proba = tree.predict_proba(Xte)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_decision_tree_regressor(regression):
    X, y = regression
    from ensemble_methods_kit.utils import train_test_split

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    tree = DecisionTree(criterion="variance", max_depth=6, random_state=0).fit(Xtr, ytr)
    assert r2_score(yte, tree.predict(Xte)) > 0.7
def test_clone_estimator_returns_unfitted_copy():
    tree = DecisionTree(max_depth=4, criterion="entropy", random_state=5)
    tree.fit(np.array([[0.0, 1.0], [2.0, 3.0]]), np.array([0, 1]))
    clone = clone_estimator(tree)
    assert clone is not tree
    assert clone.max_depth == 4
    assert clone.criterion == "entropy"
    # the clone must not carry fitted state
    assert clone._tree is None


def test_clone_then_fit_is_independent():
    tree = DecisionTree(max_depth=3, random_state=2)
    clone = clone_estimator(tree)
    clone.fit(np.array([[0.0, 1.0], [2.0, 3.0], [1.0, 0.0], [3.0, 4.0], [1.0, 2.0]]), np.array([0, 1, 0, 1, 0]))
    assert clone._tree is not None
    assert tree._tree is None
