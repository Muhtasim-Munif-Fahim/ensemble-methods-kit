"""Tests for Caruana ensemble selection."""

from __future__ import annotations

import numpy as np
import pytest

from ensemble_methods_kit import (
    DecisionTree,
    EnsembleSelectionClassifier,
    EnsembleSelectionRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    RidgeRegression,
    accuracy_score,
    train_test_split,
)


class _ConstantProba:
    """Predicts a fixed probability vector regardless of X."""

    def __init__(self, p=0.5):
        self.p = p

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        n = np.asarray(X).shape[0]
        return np.tile([1.0 - self.p, self.p], (n, 1))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class _ConstantReg:
    def __init__(self, value=0.0):
        self.value = value

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(np.asarray(X).shape[0], float(self.value))


class _Reversed:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth

    def fit(self, X, y):
        self._t = DecisionTree(max_depth=self.max_depth, random_state=0).fit(X, y)
        self.classes_ = self._t.classes_[::-1].copy()
        return self

    def predict_proba(self, X):
        return np.asarray(self._t.predict_proba(X))[:, ::-1]

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def _library():
    return [
        ("rf", RandomForestClassifier(n_estimators=10, max_depth=4, random_state=0)),
        ("tree", DecisionTree(max_depth=3, random_state=0)),
        ("coin", _ConstantProba(0.5)),
    ]


def test_classifier_accuracy_and_weights(binary_cls):
    X, y = binary_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    sel = EnsembleSelectionClassifier(_library(), n_iterations=15, random_state=0).fit(Xtr, ytr)
    assert accuracy_score(yte, sel.predict(Xte)) > 0.9
    assert np.isclose(sel.weights_.sum(), 1.0)
    assert sel.named_weights_["coin"] == 0.0
    proba = sel.predict_proba(Xte)
    assert proba.shape == (len(yte), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_weights_match_selection_counts(multiclass_cls):
    X, y = multiclass_cls
    sel = EnsembleSelectionClassifier(_library()[:2], n_iterations=7, use_best=False,
                                      random_state=0).fit(X, y)
    assert len(sel.selection_order_) == 1 + 7
    counts = np.bincount(sel.selection_order_, minlength=2)
    assert np.array_equal(counts, sel.selection_counts_)
    assert np.allclose(sel.weights_, counts / counts.sum())
    assert sel.validation_scores_.shape == (8,)
    assert sel.classes_.tolist() == [0, 1, 2]


def test_use_best_truncates_at_minimum():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] > 0).astype(int)
    sel = EnsembleSelectionClassifier(
        [("good", DecisionTree(max_depth=2, random_state=0)), ("coin", _ConstantProba(0.5))],
        n_iterations=10, metric="log_loss", random_state=0,
    ).fit(X, y)
    assert np.isclose(sel.best_score_, sel.validation_scores_.min())
    assert len(sel.selection_order_) == 1 + int(np.argmin(sel.validation_scores_))


def test_greedy_choice_hand_computed():
    """With a perfect and a useless regressor the greedy path keeps the perfect one."""
    X = np.arange(40, dtype=float).reshape(-1, 1)
    y = np.zeros(40)
    sel = EnsembleSelectionRegressor(
        [("zero", _ConstantReg(0.0)), ("off", _ConstantReg(3.0))],
        n_iterations=5, use_best=False, random_state=0,
    ).fit(X, y)
    assert sel.selection_order_ == [0] * 6
    assert np.allclose(sel.weights_, [1.0, 0.0])
    assert np.allclose(sel.predict(X), 0.0)


def test_regressor_mixes_biased_models():
    """Two models biased in opposite directions should be averaged."""
    X = np.arange(40, dtype=float).reshape(-1, 1)
    y = np.full(40, 1.0)
    sel = EnsembleSelectionRegressor(
        [("low", _ConstantReg(0.0)), ("high", _ConstantReg(2.0)), ("far", _ConstantReg(9.0))],
        n_iterations=9, use_best=True, random_state=0,
    ).fit(X, y)
    assert np.isclose(sel.weights_[0], sel.weights_[1])
    assert sel.weights_[2] == 0.0
    assert np.allclose(sel.predict(X[:3]), 1.0)
    assert np.isclose(sel.best_score_, 0.0)


def test_regressor_real_models(regression):
    X, y = regression
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    sel = EnsembleSelectionRegressor(
        [
            ("ridge", RidgeRegression(alpha=0.1)),
            ("rf", RandomForestRegressor(n_estimators=10, max_depth=4, random_state=0)),
            ("const", _ConstantReg(0.0)),
        ],
        n_iterations=10, metric="mae", random_state=0,
    ).fit(Xtr, ytr)
    assert sel.score(Xte, yte) > 0.8
    assert sel.named_weights_["ridge"] > sel.named_weights_["const"]


def test_n_init_sorted_initialisation(binary_cls):
    X, y = binary_cls
    sel = EnsembleSelectionClassifier(_library(), n_iterations=0, n_init=2,
                                      random_state=0).fit(X, y)
    assert len(sel.selection_order_) == 2
    assert 2 not in sel.selection_order_  # the coin flip is the worst single model


def test_class_order_alignment_and_string_labels(binary_cls):
    X, y = binary_cls
    labels = np.where(y == 1, "yes", "no")
    sel = EnsembleSelectionClassifier(
        [("rev", _Reversed()), ("tree", DecisionTree(max_depth=3, random_state=0))],
        n_iterations=4, metric="brier", random_state=0,
    ).fit(X, labels)
    assert set(sel.predict(X)) <= {"yes", "no"}
    assert accuracy_score(labels, sel.predict(X)) > 0.9


def test_no_refit_keeps_training_split_models(binary_cls):
    X, y = binary_cls
    sel = EnsembleSelectionClassifier(_library(), n_iterations=3, refit=False,
                                      metric="error", random_state=0).fit(X, y)
    assert len(sel.estimators_) == 3
    assert accuracy_score(y, sel.predict(X)) > 0.9


def test_inputs_are_not_fitted_and_validation(binary_cls):
    X, y = binary_cls
    tree = DecisionTree(max_depth=2, random_state=0)
    sel = EnsembleSelectionClassifier([("t", tree)], n_iterations=2, random_state=0).fit(X, y)
    assert tree.classes_ is None
    with pytest.raises(ValueError, match="features"):
        sel.predict(X[:, :1])
    with pytest.raises(RuntimeError, match="not fitted"):
        EnsembleSelectionClassifier([("t", tree)]).predict(X)
    with pytest.raises(ValueError):
        EnsembleSelectionClassifier([])
    with pytest.raises(ValueError, match="unique"):
        EnsembleSelectionClassifier([("a", tree), ("a", tree)])
    with pytest.raises(ValueError, match="metric"):
        EnsembleSelectionClassifier([("t", tree)], metric="mse")
    with pytest.raises(ValueError, match="n_init"):
        EnsembleSelectionClassifier([("t", tree)], n_init=2)
    with pytest.raises(ValueError, match="validation_fraction"):
        EnsembleSelectionRegressor([("c", _ConstantReg())], validation_fraction=1.0)
    with pytest.raises(ValueError, match="predict_proba"):
        EnsembleSelectionClassifier([("c", _ConstantReg())])
    with pytest.raises(ValueError, match="2 classes"):
        EnsembleSelectionClassifier([("t", tree)]).fit(X, np.zeros(len(y)))
