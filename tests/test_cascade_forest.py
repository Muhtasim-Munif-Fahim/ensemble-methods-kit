"""Tests for the gcForest-style CascadeForestClassifier."""

from __future__ import annotations

import numpy as np
import pytest

from ensemble_methods_kit import CascadeForestClassifier, accuracy_score
from ensemble_methods_kit.cascade_forest import _stratified_folds
from ensemble_methods_kit.utils import train_test_split


def _noisy_xor(n=360, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 4))
    y = ((X[:, 0] * X[:, 1]) > 0).astype(int)
    return X, y


def test_multiclass_accuracy_and_proba(multiclass_cls):
    X, y = multiclass_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    m = CascadeForestClassifier(n_estimators=10, random_state=0).fit(Xtr, ytr)
    assert accuracy_score(yte, m.predict(Xte)) > 0.9
    p = m.predict_proba(Xte)
    assert p.shape == (len(yte), 3)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.all(p >= 0)


def test_layer_growth_and_truncation_to_best():
    X, y = _noisy_xor()
    m = CascadeForestClassifier(
        n_estimators=10, max_layers=4, early_stopping_rounds=2, random_state=1
    ).fit(X, y)
    scores = m.layer_scores_
    assert 1 <= len(scores) <= 4
    # The kept depth is the first layer reaching the best out-of-fold score.
    assert m.best_layer_ == int(np.argmax(scores)) + 1
    assert m.n_layers_ == len(m.layers_) == m.best_layer_
    # Each layer holds one RF + one completely random forest per pair.
    assert all(len(layer) == 2 for layer in m.layers_)
    assert len(m.layer_proba(X[:5])) == m.best_layer_


def test_tol_forces_single_layer_and_max_layers_cap():
    X, y = _noisy_xor()
    m = CascadeForestClassifier(n_estimators=5, tol=1.0, random_state=0).fit(X, y)
    assert m.best_layer_ == 1 and len(m.layer_scores_) == 2
    m2 = CascadeForestClassifier(
        n_estimators=5, max_layers=3, early_stopping_rounds=10, n_forests=2, random_state=0
    ).fit(X, y)
    assert len(m2.layer_scores_) == 3
    assert all(len(layer) == 4 for layer in m2.layers_)


def test_xor_learned_better_than_chance():
    X, y = _noisy_xor(400, seed=3)
    m = CascadeForestClassifier(n_estimators=15, random_state=0).fit(X[:300], y[:300])
    assert m.score(X[300:], y[300:]) > 0.9


def test_string_labels_and_determinism():
    X, y = _noisy_xor(240, seed=4)
    labels = np.array(["neg", "pos"])[y]
    a = CascadeForestClassifier(n_estimators=6, random_state=7).fit(X, labels)
    b = CascadeForestClassifier(n_estimators=6, random_state=7).fit(X, labels)
    assert set(a.predict(X)) <= {"neg", "pos"}
    assert list(a.classes_) == ["neg", "pos"]
    np.testing.assert_allclose(a.predict_proba(X), b.predict_proba(X))
    assert a.layer_scores_ == b.layer_scores_


def test_stratified_folds_balanced():
    y = np.r_[np.zeros(9, int), np.ones(6, int)]
    folds = _stratified_folds(y, 3, np.random.default_rng(0))
    for f in range(3):
        assert np.sum((folds == f) & (y == 0)) == 3
        assert np.sum((folds == f) & (y == 1)) == 2


def test_validation_errors():
    X, y = _noisy_xor(60)
    with pytest.raises(RuntimeError):
        CascadeForestClassifier().predict(X)
    with pytest.raises(ValueError):
        CascadeForestClassifier(n_folds=1).fit(X, y)
    with pytest.raises(ValueError):
        CascadeForestClassifier(max_layers=0).fit(X, y)
    with pytest.raises(ValueError):
        CascadeForestClassifier(tol=-0.1).fit(X, y)
    with pytest.raises(ValueError):
        CascadeForestClassifier().fit(X, np.zeros(60))
    with pytest.raises(ValueError):
        CascadeForestClassifier().fit(X, y[:-1])
    with pytest.raises(ValueError):
        CascadeForestClassifier(n_folds=3).fit(X[:4], np.array([0, 0, 1, 1]))
