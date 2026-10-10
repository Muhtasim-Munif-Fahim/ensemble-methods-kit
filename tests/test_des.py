"""Tests for KNORA-E / KNORA-U dynamic ensemble selection."""

from __future__ import annotations

import numpy as np
import pytest

from ensemble_methods_kit import (
    DecisionTree,
    KNORAClassifier,
    accuracy_score,
    train_test_split,
)


class _Constant:
    """Always predicts one fixed label."""

    def __init__(self, label=0):
        self.label = label

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(np.asarray(X).shape[0], self.label)


class _Flipped:
    """A decision tree whose binary predictions are inverted."""

    def __init__(self, max_depth=3):
        self.max_depth = max_depth

    def fit(self, X, y):
        self._t = DecisionTree(max_depth=self.max_depth, random_state=0).fit(X, y)
        return self

    def predict(self, X):
        return 1 - np.asarray(self._t.predict(X))


def _xor(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    return X, y


@pytest.mark.parametrize("method", ["eliminate", "union"])
def test_binary_accuracy(split_cls, method):
    X_tr, X_te, y_tr, y_te = split_cls
    clf = KNORAClassifier(method=method, n_estimators=10, random_state=0).fit(X_tr, y_tr)
    assert accuracy_score(y_te, clf.predict(X_te)) > 0.95
    assert len(clf.estimators_) == 10
    assert clf.dsel_correct_.shape[1] == 10
    assert np.all((clf.pool_accuracy_ >= 0) & (clf.pool_accuracy_ <= 1))


@pytest.mark.parametrize("method", ["eliminate", "union"])
def test_multiclass_proba(multiclass_cls, method):
    X, y = multiclass_cls
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
    clf = KNORAClassifier(method=method, random_state=1).fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)
    assert proba.shape == (X_te.shape[0], 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert np.all(proba >= 0)
    np.testing.assert_array_equal(clf.predict(X_te), clf.classes_[proba.argmax(axis=1)])
    assert accuracy_score(y_te, clf.predict(X_te)) > 0.9


def test_eliminate_drops_bad_models(split_cls):
    X_tr, X_te, y_tr, y_te = split_cls
    pool = [
        ("tree", DecisionTree(max_depth=3, random_state=0)),
        ("flipped", _Flipped()),
        ("zero", _Constant(0)),
    ]
    clf = KNORAClassifier(estimators=pool, method="eliminate", k=5, random_state=0)
    clf.fit(X_tr, y_tr)
    w = clf.competence_weights(X_te)
    assert set(np.unique(w)) <= {0.0, 1.0}
    # the inverted model is never an oracle where the tree is accurate
    assert w[:, 1].mean() < 0.1
    assert accuracy_score(y_te, clf.predict(X_te)) > 0.95


def test_eliminate_selected_models_are_local_oracles(split_cls):
    X_tr, X_te, y_tr, _ = split_cls
    clf = KNORAClassifier(n_estimators=8, max_depth=1, k=4, random_state=3).fit(X_tr, y_tr)
    w = clf.competence_weights(X_te)
    nbrs = clf._neighbours(X_te)
    for i in range(X_te.shape[0]):
        sel = w[i] > 0
        correct = clf.dsel_correct_[nbrs[i]]
        if np.all(correct, axis=0).any():
            # with a full-k oracle available, selection equals the full-k oracle set
            np.testing.assert_array_equal(sel, np.all(correct, axis=0))
        assert sel.any()


def test_union_weights_are_correct_neighbour_counts(split_cls):
    X_tr, X_te, y_tr, _ = split_cls
    clf = KNORAClassifier(method="union", n_estimators=6, k=5, random_state=2).fit(X_tr, y_tr)
    w = clf.competence_weights(X_te)
    nbrs = clf._neighbours(X_te)
    expected = clf.dsel_correct_[nbrs].sum(axis=1).astype(float)
    expected[expected.sum(axis=1) == 0] = 1.0
    np.testing.assert_allclose(w, expected)
    assert w.max() <= 5


def test_local_selection_beats_static_vote_on_xor():
    X, y = _xor(600, seed=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    clf = KNORAClassifier(n_estimators=25, max_depth=1, k=7, random_state=0).fit(X_tr, y_tr)
    preds = np.column_stack([est.predict(X_te) for est in clf.estimators_])
    static = (preds.mean(axis=1) >= 0.5).astype(int)
    knora = accuracy_score(y_te, clf.predict(X_te))
    assert knora > accuracy_score(y_te, static) + 0.1
    assert np.all(clf.n_selected(X_te) >= 1)


def test_deterministic(split_cls):
    X_tr, X_te, y_tr, _ = split_cls
    a = KNORAClassifier(random_state=5).fit(X_tr, y_tr).predict_proba(X_te)
    b = KNORAClassifier(random_state=5).fit(X_tr, y_tr).predict_proba(X_te)
    np.testing.assert_array_equal(a, b)


def test_k_larger_than_dsel_is_clipped(binary_cls):
    X, y = binary_cls
    clf = KNORAClassifier(k=10_000, random_state=0).fit(X[:40], y[:40])
    assert clf.predict(X[40:50]).shape == (10,)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "oracle"},
        {"k": 0},
        {"dsel_fraction": 0.0},
        {"dsel_fraction": 1.0},
        {"n_estimators": 0},
        {"estimators": []},
        {"estimators": [("a", DecisionTree()), ("a", DecisionTree())]},
        {"estimators": [("a", object())]},
    ],
)
def test_invalid_params(kwargs):
    with pytest.raises(ValueError):
        KNORAClassifier(**kwargs)


def test_errors(split_cls):
    X_tr, X_te, y_tr, _ = split_cls
    with pytest.raises(RuntimeError):
        KNORAClassifier().predict(X_te)
    clf = KNORAClassifier(random_state=0).fit(X_tr, y_tr)
    with pytest.raises(ValueError):
        clf.predict(X_te[:, :2])
    with pytest.raises(ValueError):
        KNORAClassifier().fit(X_tr, np.zeros(X_tr.shape[0]))
