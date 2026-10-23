"""Tests for LogitBoost, Gentle AdaBoost and the weighted regression tree."""

from __future__ import annotations

import numpy as np
import pytest

from ensemble_methods_kit import (
    GentleBoostClassifier,
    LogitBoostClassifier,
    WeightedRegressionTree,
    accuracy_score,
)


def _hastie(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 10))
    y = np.where((X ** 2).sum(axis=1) > 9.34, 1, -1)
    return X, y


class TestWeightedRegressionTree:
    def test_stump_matches_brute_force(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 3))
        z = rng.normal(size=40)
        w = rng.random(40)
        best = None
        for j in range(3):
            for thr in np.unique(X[:, j])[:-1]:
                left = X[:, j] <= thr
                sse = 0.0
                for m in (left, ~left):
                    mu = np.dot(w[m], z[m]) / w[m].sum()
                    sse += np.dot(w[m], (z[m] - mu) ** 2)
                if best is None or sse < best[0] - 1e-12:
                    best = (sse, j, thr)
        tree = WeightedRegressionTree(max_depth=1).fit(X, z, w)
        pred = tree.predict(X)
        assert np.dot(w, (z - pred) ** 2) == pytest.approx(best[0])
        assert tree._nodes[0][1] == best[1]

    def test_leaf_is_weighted_mean_and_respects_weights(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        z = np.array([1.0, 3.0, 10.0, 20.0])
        w = np.array([3.0, 1.0, 1.0, 1.0])
        tree = WeightedRegressionTree(max_depth=1).fit(X, z, w)
        pred = tree.predict(X)
        assert pred[0] == pytest.approx(pred[1])
        assert pred[0] == pytest.approx((3 * 1 + 3) / 4)
        assert pred[2] == pytest.approx(15.0)

    def test_zero_weight_rows_are_ignored(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        z = np.array([0.0, 0.0, 5.0, 100.0])
        tree = WeightedRegressionTree(max_depth=2).fit(X, z, [1, 1, 1, 0])
        assert tree.predict([[2.0]])[0] == pytest.approx(5.0)

    def test_min_samples_leaf_and_depth(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(30, 2))
        z = rng.normal(size=30)
        tree = WeightedRegressionTree(max_depth=3, min_samples_leaf=8).fit(X, z)
        leaves = tree.predict(X)
        _, counts = np.unique(leaves, return_counts=True)
        assert counts.min() >= 8 and len(counts) <= 3

    def test_validation(self):
        with pytest.raises(ValueError):
            WeightedRegressionTree(max_depth=0)
        with pytest.raises(ValueError):
            WeightedRegressionTree().fit(np.zeros((3, 1)), [1, 2, 3], [0, 0, 0])
        with pytest.raises(ValueError):
            WeightedRegressionTree().fit(np.zeros((3, 1)), [1, 2])
        with pytest.raises(RuntimeError):
            WeightedRegressionTree().predict([[1.0]])


class TestLogitBoost:
    def test_binary_reduces_to_friedman_algorithm_3(self):
        X, y = _hastie(300, 0)
        ys = (y == 1).astype(float)
        F = np.zeros(300)
        for _ in range(15):
            p = 1.0 / (1.0 + np.exp(-2.0 * F))
            w = np.maximum(p * (1 - p), 1e-10)
            z = np.clip((ys - p) / w, -4, 4)
            F += 0.5 * WeightedRegressionTree(1).fit(X, z, w).predict(X)
        model = LogitBoostClassifier(n_estimators=15, learning_rate=1.0).fit(X, y)
        assert np.allclose(model.decision_function(X)[:, 1], F)
        assert np.allclose(model.predict_proba(X)[:, 1], 1 / (1 + np.exp(-2 * F)))

    def test_hastie_accuracy(self):
        X, y = _hastie(2000, 1)
        Xt, yt = _hastie(1000, 2)
        model = LogitBoostClassifier(n_estimators=150, learning_rate=0.5).fit(X, y)
        assert accuracy_score(yt, model.predict(Xt)) > 0.88
        stump = LogitBoostClassifier(n_estimators=1).fit(X, y)
        assert accuracy_score(yt, model.predict(Xt)) > accuracy_score(yt, stump.predict(Xt)) + 0.2

    def test_training_loss_decreases(self):
        X, y = _hastie(800, 3)
        model = LogitBoostClassifier(n_estimators=60, learning_rate=0.3).fit(X, y)
        loss = np.array(model.train_loss_)
        assert loss[0] == pytest.approx(np.log(2))
        assert len(loss) == 61
        assert np.all(np.diff(loss) < 1e-9)

    def test_multiclass(self, multiclass_cls):
        X, y = multiclass_cls
        model = LogitBoostClassifier(n_estimators=40, learning_rate=0.5).fit(X, y)
        P = model.predict_proba(X)
        assert P.shape == (len(y), 3)
        assert np.allclose(P.sum(axis=1), 1.0)
        assert np.allclose(model.decision_function(X).sum(axis=1), 0.0)
        assert accuracy_score(y, model.predict(X)) > 0.97
        assert len(model.estimators_[0]) == 3

    def test_staged_predict_and_string_labels(self, binary_cls):
        X, y = binary_cls
        labels = np.where(y == 1, "spam", "ham")
        model = LogitBoostClassifier(n_estimators=10, max_depth=2).fit(X, labels)
        stages = list(model.staged_predict(X))
        assert len(stages) == 10
        assert np.array_equal(stages[-1], model.predict(X))
        assert set(model.predict(X)) <= {"spam", "ham"}

    def test_validation(self):
        for kwargs in ({"n_estimators": 0}, {"learning_rate": 0}, {"max_depth": 0}, {"z_max": 0}):
            with pytest.raises(ValueError):
                LogitBoostClassifier(**kwargs)
        with pytest.raises(ValueError):
            LogitBoostClassifier().fit(np.zeros((4, 2)), [1, 1, 1, 1])
        with pytest.raises(RuntimeError):
            LogitBoostClassifier().predict(np.zeros((2, 2)))


class TestGentleBoost:
    def test_hastie_accuracy(self):
        X, y = _hastie(2000, 4)
        Xt, yt = _hastie(1000, 5)
        model = GentleBoostClassifier(n_estimators=150, learning_rate=0.5).fit(X, y)
        assert accuracy_score(yt, model.predict(Xt)) > 0.86

    def test_matches_manual_gentle_adaboost(self):
        X, y = _hastie(200, 6)
        F = np.zeros(200)
        for _ in range(10):
            w = np.exp(-y * F)
            F += WeightedRegressionTree(1).fit(X, y.astype(float), w / w.sum()).predict(X)
        model = GentleBoostClassifier(n_estimators=10).fit(X, y)
        assert np.allclose(model.decision_function(X), F)
        assert np.allclose(model.predict_proba(X)[:, 1], 1 / (1 + np.exp(-2 * F)))

    def test_exponential_loss_decreases(self):
        X, y = _hastie(600, 7)
        model = GentleBoostClassifier(n_estimators=40).fit(X, y)
        assert np.all(np.diff(model.train_loss_) < 1e-9)

    def test_xor_needs_depth_two(self):
        rng = np.random.default_rng(8)
        X = rng.uniform(-1, 1, size=(400, 2))
        y = (X[:, 0] * X[:, 1] > 0).astype(int)
        deep = GentleBoostClassifier(n_estimators=30, max_depth=2).fit(X, y)
        assert accuracy_score(y, deep.predict(X)) > 0.95

    def test_multiclass_one_vs_rest(self, multiclass_cls):
        X, y = multiclass_cls
        model = GentleBoostClassifier(n_estimators=30).fit(X, y)
        P = model.predict_proba(X)
        assert np.allclose(P.sum(axis=1), 1.0)
        assert model.decision_function(X).shape == (len(y), 3)
        assert accuracy_score(y, model.predict(X)) > 0.97
        assert len(list(model.staged_predict(X))) == 30

    def test_validation(self):
        with pytest.raises(ValueError):
            GentleBoostClassifier(learning_rate=-1)
        with pytest.raises(ValueError):
            GentleBoostClassifier().fit(np.zeros((3, 2)), [0, 1])
        with pytest.raises(RuntimeError):
            GentleBoostClassifier().predict_proba(np.zeros((2, 2)))
