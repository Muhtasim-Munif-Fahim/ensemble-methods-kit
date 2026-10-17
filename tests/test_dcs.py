"""Tests for OLA / LCA dynamic classifier selection (Woods et al. 1997)."""

from __future__ import annotations

import numpy as np
import pytest

from ensemble_methods_kit import (
    DCSClassifier,
    DecisionTree,
    KNORAClassifier,
    accuracy_score,
    train_test_split,
)


class _Constant:
    def __init__(self, label=0):
        self.label = label

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(np.asarray(X).shape[0], self.label)


class _Threshold:
    """Predicts 1 when feature 0 exceeds ``t``."""

    def __init__(self, t=0.0):
        self.t = t

    def fit(self, X, y):
        return self

    def predict(self, X):
        return (np.asarray(X)[:, 0] > self.t).astype(int)


def _line(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 1))
    y = (X[:, 0] > 0).astype(int)
    return X, y


@pytest.mark.parametrize("method", ["ola", "lca"])
def test_accuracy_on_separable_blobs(binary_cls, method):
    X, y = binary_cls
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    clf = DCSClassifier(method=method, n_estimators=9, random_state=0).fit(X_tr, y_tr)
    assert accuracy_score(y_te, clf.predict(X_te)) > 0.95
    proba = clf.predict_proba(X_te)
    assert proba.shape == (X_te.shape[0], 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


@pytest.mark.parametrize("method", ["ola", "lca"])
def test_local_selection_picks_the_locally_correct_constant(method):
    X, y = _line()
    pool = [("zero", _Constant(0)), ("one", _Constant(1))]
    clf = DCSClassifier(estimators=pool, method=method, k=5, random_state=0).fit(X, y)
    queries = np.array([[-0.8], [-0.5], [0.5], [0.8]])
    assert clf.selected_estimator(queries).tolist() == [0, 0, 1, 1]
    assert clf.predict(queries).tolist() == [0, 0, 1, 1]
    # a static vote of the two constants could never do this
    assert np.all(clf.n_selected(queries) == 1)


def test_ola_competence_is_local_accuracy():
    X, y = _line(seed=1)
    pool = [("t0", _Threshold(0.0)), ("t5", _Threshold(0.5)), ("zero", _Constant(0))]
    clf = DCSClassifier(estimators=pool, method="ola", k=6, random_state=3).fit(X, y)
    Xq = np.random.default_rng(9).uniform(-1, 1, (25, 1))
    comp = clf.competences(Xq)
    nbrs = clf._neighbours(Xq)
    expected = clf.dsel_correct_[nbrs].mean(axis=1)
    assert np.allclose(comp, expected)
    assert np.all(comp[:, 0] == 1.0)  # the true boundary is always right


def test_lca_matches_brute_force_definition():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(240, 3))
    y = (X[:, 0] + 0.6 * X[:, 1] + 0.4 * rng.normal(size=240) > 0).astype(int)
    clf = DCSClassifier(method="lca", n_estimators=6, max_depth=2, k=7, random_state=1).fit(X, y)
    Xq = rng.normal(size=(30, 3))
    comp = clf.competences(Xq)
    nbrs = clf._neighbours(Xq)
    for i in range(Xq.shape[0]):
        for m, est in enumerate(clf.estimators_):
            w = est.predict(Xq[i : i + 1])[0]
            labelled_w = [j for j in nbrs[i] if clf._dsel_pred[j, m] == w]
            if not labelled_w:
                assert comp[i, m] == 0.0
            else:
                hits = sum(clf._dsel_y[j] == w for j in labelled_w)
                assert comp[i, m] == pytest.approx(hits / len(labelled_w))


def test_constant_member_competence_is_local_class_share():
    # For "always 1" both OLA accuracy and LCA precision for label 1 reduce
    # to the share of class-1 points in the region of competence.
    X, y = _line(seed=2)
    pool = [("one", _Constant(1)), ("t0", _Threshold(0.0))]
    ola = DCSClassifier(estimators=pool, method="ola", k=9, random_state=0).fit(X, y)
    lca = DCSClassifier(estimators=pool, method="lca", k=9, random_state=0).fit(X, y)
    q = np.array([[0.02]])
    assert ola.competences(q)[0, 1] == 1.0
    assert lca.competences(q)[0, 1] == 1.0
    assert ola.competences(q)[0, 0] == pytest.approx(lca.competences(q)[0, 0])


def test_ties_vote_together():
    X, y = _line(seed=5)
    pool = [("a", _Threshold(0.0)), ("b", _Threshold(0.0)), ("zero", _Constant(0))]
    clf = DCSClassifier(estimators=pool, method="ola", k=5, random_state=0).fit(X, y)
    q = np.array([[0.7]])
    weights = clf.competence_weights(q)
    assert weights.tolist() == [[1.0, 1.0, 0.0]]
    assert clf.n_selected(q).tolist() == [2]
    assert clf.selected_estimator(q).tolist() == [0]


def test_multiclass(multiclass_cls):
    X, y = multiclass_cls
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    clf = DCSClassifier(method="lca", n_estimators=8, random_state=2).fit(X_tr, y_tr)
    assert accuracy_score(y_te, clf.predict(X_te)) > 0.9
    assert set(np.unique(clf.predict(X_te))) <= set(clf.classes_)


def test_validation_and_unfitted():
    with pytest.raises(ValueError, match="method"):
        DCSClassifier(method="eliminate")
    with pytest.raises(ValueError, match="method"):
        KNORAClassifier(method="ola")
    with pytest.raises(ValueError, match="tie_tolerance"):
        DCSClassifier(tie_tolerance=-1.0)
    with pytest.raises(RuntimeError):
        DCSClassifier().predict(np.zeros((2, 2)))
    X, y = _line()
    clf = DCSClassifier(n_estimators=3, random_state=0).fit(X, y)
    with pytest.raises(ValueError, match="features"):
        clf.predict(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="DCSClassifier needs at least two classes"):
        DCSClassifier().fit(X, np.zeros(X.shape[0], dtype=int))


def test_reproducible():
    X, y = _line(seed=7)
    a = DCSClassifier(n_estimators=5, random_state=11).fit(X, y)
    b = DCSClassifier(n_estimators=5, random_state=11).fit(X, y)
    Xq = np.linspace(-1, 1, 21).reshape(-1, 1)
    assert np.array_equal(a.predict(Xq), b.predict(Xq))
    assert np.allclose(a.competences(Xq), b.competences(Xq))
