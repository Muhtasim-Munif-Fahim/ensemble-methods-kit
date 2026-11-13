"""Tests for SMOTE oversampling and SMOTEBoost."""

from __future__ import annotations

import numpy as np
import pytest

from ensemble_methods_kit import AdaBoostClassifier, SMOTEBoostClassifier, recall_score, smote


def _imbalanced(seed=0, n_maj=200, n_min=20, d=4, shift=2.0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(0.0, 1.0, (n_maj, d)), rng.normal(shift, 1.0, (n_min, d))])
    y = np.r_[np.zeros(n_maj, dtype=int), np.ones(n_min, dtype=int)]
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def test_smote_auto_balances_classes():
    X, y = _imbalanced()
    Xr, yr = smote(X, y, random_state=0)
    assert Xr.shape[0] == 400
    np.testing.assert_array_equal(np.bincount(yr), [200, 200])
    np.testing.assert_array_equal(Xr[: len(X)], X)


def test_synthetic_points_lie_on_segments_between_minority_neighbours():
    X, y = _imbalanced(n_min=12)
    X_min = X[y == 1]
    Xs, ys = smote(X, y, k_neighbors=3, random_state=1, return_synthetic_only=True)
    assert np.all(ys == 1)
    for p in Xs:
        ok = False
        for a in range(len(X_min)):
            for b in range(len(X_min)):
                if a == b:
                    continue
                v = X_min[b] - X_min[a]
                t = float(np.dot(p - X_min[a], v) / np.dot(v, v))
                if -1e-9 <= t <= 1 + 1e-9 and np.allclose(X_min[a] + t * v, p, atol=1e-8):
                    ok = True
                    break
            if ok:
                break
        assert ok
    # Synthetic points stay inside the minority bounding box.
    assert np.all(Xs >= X_min.min(axis=0) - 1e-12)
    assert np.all(Xs <= X_min.max(axis=0) + 1e-12)


def test_smote_neighbour_restriction_k1():
    X_min = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0], [10.0, 11.0]])
    X = np.vstack([np.zeros((10, 2)) + 5.0, X_min])
    y = np.r_[np.zeros(10, dtype=int), np.ones(4, dtype=int)]
    Xs, _ = smote(X, y, k_neighbors=1, random_state=0, return_synthetic_only=True)
    # With k=1 each synthetic point lies between a pair of nearest neighbours.
    near_a = (Xs[:, 0] <= 1.0) & (Xs[:, 1] == 0.0)
    near_b = (Xs[:, 0] == 10.0) & (Xs[:, 1] >= 10.0)
    assert np.all(near_a | near_b)


def test_smote_float_strategy_and_multiclass():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(130, 3))
    y = np.r_[np.zeros(100, int), np.ones(20, int), np.full(10, 2)]
    _, ys = smote(X, y, sampling_strategy=2.0, random_state=0, return_synthetic_only=True)
    np.testing.assert_array_equal(np.bincount(ys, minlength=3), [0, 40, 20])
    _, yr = smote(X, y, random_state=0)
    np.testing.assert_array_equal(np.bincount(yr), [100, 100, 100])


def test_smote_single_minority_point_and_determinism():
    X = np.r_[np.zeros((5, 2)), [[3.0, 4.0]]]
    y = np.r_[np.zeros(5, int), [1]]
    Xs, _ = smote(X, y, return_synthetic_only=True, random_state=0)
    np.testing.assert_allclose(Xs, np.tile([3.0, 4.0], (4, 1)))
    Xa, _ = _imbalanced()
    a = smote(Xa, _imbalanced()[1], random_state=7)[0]
    b = smote(Xa, _imbalanced()[1], random_state=7)[0]
    np.testing.assert_allclose(a, b)


def test_smote_validation():
    X, y = _imbalanced()
    with pytest.raises(ValueError):
        smote(X, y, k_neighbors=0)
    with pytest.raises(ValueError):
        smote(X, y, sampling_strategy="minority")
    with pytest.raises(ValueError):
        smote(X, y, sampling_strategy=-1.0)
    with pytest.raises(ValueError):
        smote(X, np.zeros(len(y)))
    with pytest.raises(ValueError):
        smote(X[:5], y)


def test_smoteboost_improves_minority_recall_over_adaboost():
    X, y = _imbalanced(seed=4, n_maj=400, n_min=30, shift=1.5)
    Xte, yte = _imbalanced(seed=5, n_maj=400, n_min=60, shift=1.5)
    base = AdaBoostClassifier(n_estimators=30, max_depth=1, random_state=0).fit(X, y)
    sb = SMOTEBoostClassifier(n_estimators=30, max_depth=1, random_state=0).fit(X, y)
    r_base = recall_score(yte, base.predict(Xte))
    r_sb = recall_score(yte, sb.predict(Xte))
    assert r_sb > r_base
    assert r_sb > 0.6


def test_smoteboost_attributes_and_proba():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (90, 2)), rng.normal(3, 1, (25, 2)), rng.normal([0, 4], 1, (12, 2))])
    y = np.r_[np.zeros(90, int), np.ones(25, int), np.full(12, 2)]
    clf = SMOTEBoostClassifier(n_estimators=15, max_depth=2, random_state=1).fit(X, y)
    assert 1 <= len(clf.estimators_) <= 15
    assert len(clf.estimator_weights_) == len(clf.estimators_) == len(clf.estimator_errors_)
    P = clf.predict_proba(X)
    np.testing.assert_allclose(P.sum(axis=1), 1.0)
    np.testing.assert_array_equal(clf.classes_[P.argmax(axis=1)], clf.predict(X))
    assert np.mean(clf.predict(X) == y) > 0.8
    a = SMOTEBoostClassifier(n_estimators=5, random_state=3).fit(X, y).predict(X)
    b = SMOTEBoostClassifier(n_estimators=5, random_state=3).fit(X, y).predict(X)
    np.testing.assert_array_equal(a, b)


def test_smoteboost_validation():
    with pytest.raises(ValueError):
        SMOTEBoostClassifier(n_estimators=0)
    with pytest.raises(ValueError):
        SMOTEBoostClassifier(learning_rate=0)
    with pytest.raises(ValueError):
        SMOTEBoostClassifier(k_neighbors=0)
    with pytest.raises(ValueError):
        SMOTEBoostClassifier(sampling_strategy="x")
    with pytest.raises(RuntimeError):
        SMOTEBoostClassifier().predict(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        SMOTEBoostClassifier().fit(np.zeros((4, 2)), np.zeros(4))
