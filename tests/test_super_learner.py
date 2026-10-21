"""Tests for the Super Learner (van der Laan, Polley & Hubbard, 2007)."""

from __future__ import annotations

import numpy as np
import pytest

from ensemble_methods_kit import (
    DecisionTree,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    RidgeRegression,
    SuperLearnerClassifier,
    SuperLearnerRegressor,
    mean_squared_error,
    nnls,
    project_to_simplex,
)
from ensemble_methods_kit.stacking import LogisticRegression, _kfold_indices
from ensemble_methods_kit.utils import train_test_split


class _Constant:
    """Predicts a constant (a deliberately bad candidate)."""

    def __init__(self, value=0.0):
        self.value = value

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(np.asarray(X).shape[0], self.value)


def _friedman(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, 5))
    y = 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2 + 10 * X[:, 3] + 5 * X[:, 4]
    return X, y + rng.normal(0, 1.0, n)


def _reg_bases():
    return [
        ("ridge", RidgeRegression(alpha=1.0)),
        ("rf", RandomForestRegressor(n_estimators=15, max_depth=6, random_state=1)),
        ("gbr", GradientBoostingRegressor(n_estimators=60, learning_rate=0.1, max_depth=3, random_state=1)),
        ("const", _Constant(0.0)),
    ]


class TestSolvers:
    def test_nnls_matches_scipy(self):
        scipy_opt = pytest.importorskip("scipy.optimize")
        rng = np.random.default_rng(0)
        for _ in range(25):
            m, n = rng.integers(5, 40), rng.integers(1, 8)
            A = rng.normal(size=(m, n))
            b = rng.normal(size=m)
            ref, _ = scipy_opt.nnls(A, b)
            assert np.allclose(nnls(A, b), ref, atol=1e-8)

    def test_nnls_kkt(self):
        rng = np.random.default_rng(1)
        A = rng.normal(size=(30, 6))
        b = A @ np.array([1.0, 0.0, 2.0, 0.0, 0.5, 0.0]) + 0.1 * rng.normal(size=30)
        x = nnls(A, b)
        g = A.T @ (A @ x - b)
        assert np.all(x >= 0)
        assert np.allclose(g[x > 0], 0.0, atol=1e-8)
        assert np.all(g[x == 0] >= -1e-8)

    def test_project_to_simplex(self):
        rng = np.random.default_rng(2)
        for _ in range(50):
            v = rng.normal(size=rng.integers(1, 9)) * 3
            p = project_to_simplex(v)
            assert p.sum() == pytest.approx(1.0) and np.all(p >= 0)
            # optimality: no simplex vertex mixture is closer
            for _ in range(20):
                q = rng.dirichlet(np.ones(v.size))
                assert np.sum((p - v) ** 2) <= np.sum((q - v) ** 2) + 1e-10


class TestRegressor:
    def test_weights_on_simplex_and_bad_learner_dropped(self):
        X, y = _friedman()
        sl = SuperLearnerRegressor(_reg_bases(), cv=5, random_state=0).fit(X, y)
        assert sl.weights_.sum() == pytest.approx(1.0) and np.all(sl.weights_ >= 0)
        assert sl.weights_[sl.names_.index("const")] == 0.0
        assert "const" not in sl.fitted_
        assert sl.super_learner_cv_risk_ <= sl.cv_risk_.min() + 1e-9

    def test_cv_predictions_are_out_of_fold(self):
        X, y = _friedman(120)
        sl = SuperLearnerRegressor([("ridge", RidgeRegression(alpha=1.0))], cv=4, random_state=3).fit(X, y)
        rng = np.random.default_rng(3)
        expected = np.zeros(len(y))
        for tr, va in _kfold_indices(len(y), 4, rng):
            expected[va] = RidgeRegression(alpha=1.0).fit(X[tr], y[tr]).predict(X[va])
        assert np.allclose(sl.cv_predictions_[:, 0], expected)
        assert sl.cv_risk_[0] == pytest.approx(np.mean((expected - y) ** 2))

    def test_methods(self):
        X, y = _friedman()
        nn = SuperLearnerRegressor(_reg_bases(), method="nnls", cv=5, random_state=0).fit(X, y)
        cc = SuperLearnerRegressor(_reg_bases(), method="convex", cv=5, random_state=0).fit(X, y)
        dd = SuperLearnerRegressor(_reg_bases(), method="discrete", cv=5, random_state=0).fit(X, y)
        assert np.count_nonzero(dd.weights_) == 1 and dd.weights_[np.argmin(dd.cv_risk_)] == 1.0
        # convex LS is the exact simplex optimum: never worse than NNLS-rescaled or discrete on CV preds
        Z = cc.cv_predictions_
        risk = lambda w: np.mean((Z @ w - y) ** 2)  # noqa: E731
        assert risk(cc.weights_) <= risk(nn.weights_) + 1e-8
        assert risk(cc.weights_) <= risk(dd.weights_) + 1e-8

    def test_beats_or_matches_best_single_learner_on_test(self):
        X, y = _friedman(500, seed=4)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
        sl = SuperLearnerRegressor(_reg_bases(), cv=5, random_state=0).fit(Xtr, ytr)
        sl_mse = mean_squared_error(yte, sl.predict(Xte))
        single = []
        for _, est in _reg_bases():
            est.fit(Xtr, ytr)
            single.append(mean_squared_error(yte, est.predict(Xte)))
        assert sl_mse <= 1.05 * min(single)

    def test_trim_and_summary(self):
        X, y = _friedman()
        sl = SuperLearnerRegressor(_reg_bases(), cv=5, random_state=0, trim=0.2).fit(X, y)
        assert np.all((sl.weights_ == 0) | (sl.weights_ >= 0.2))
        rows = sl.summary()
        assert [r["weight"] for r in rows] == sorted([r["weight"] for r in rows], reverse=True)
        assert {r["name"] for r in rows} == {"ridge", "rf", "gbr", "const"}

    def test_validation(self):
        X, y = _friedman(30)
        with pytest.raises(ValueError):
            SuperLearnerRegressor([]).fit(X, y)
        with pytest.raises(ValueError):
            SuperLearnerRegressor(_reg_bases(), method="log_loss").fit(X, y)
        with pytest.raises(ValueError):
            SuperLearnerRegressor([("a", _Constant()), ("a", _Constant())]).fit(X, y)
        with pytest.raises(RuntimeError):
            SuperLearnerRegressor(_reg_bases()).predict(X)


class _BadProba:
    """Uniform probabilities (a deliberately bad classifier)."""

    def __init__(self):
        pass

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        return self

    def predict_proba(self, X):
        return np.full((np.asarray(X).shape[0], self.classes_.size), 1.0 / self.classes_.size)


def _cls_bases():
    return [
        ("logit", LogisticRegression(n_iter=300)),
        ("rf", RandomForestClassifier(n_estimators=15, max_depth=5, random_state=1)),
        ("tree", DecisionTree(max_depth=3, random_state=1)),
        ("uniform", _BadProba()),
    ]


class TestClassifier:
    @pytest.mark.parametrize("method", ["log_loss", "nnls", "convex", "discrete"])
    def test_methods_produce_valid_probabilities(self, method, multiclass_cls):
        X, y = multiclass_cls
        sl = SuperLearnerClassifier(_cls_bases(), method=method, cv=5, random_state=0).fit(X, y)
        P = sl.predict_proba(X)
        assert P.shape == (len(y), 3) and np.allclose(P.sum(axis=1), 1.0)
        assert sl.weights_.sum() == pytest.approx(1.0)
        assert np.mean(sl.predict(X) == y) > 0.9

    def test_log_loss_weights_optimal_on_cv(self, binary_cls):
        X, y = binary_cls
        rng = np.random.default_rng(0)
        y = np.where(rng.random(len(y)) < 0.1, 1 - y, y)  # label noise so learners differ
        sl = SuperLearnerClassifier(_cls_bases(), cv=5, random_state=0).fit(X, y)
        assert sl.super_learner_cv_risk_ <= sl.cv_risk_.min() + 1e-9
        P = sl.cv_predictions_
        yi = np.searchsorted(sl.classes_, y)

        def nll(w):
            return -np.mean(np.log(np.clip(np.einsum("nlk,l->nk", P, w)[np.arange(len(y)), yi], 1e-15, 1)))

        for _ in range(100):
            assert nll(sl.weights_) <= nll(rng.dirichlet(np.ones(4))) + 1e-7
        assert sl.cv_risk_[sl.names_.index("uniform")] == pytest.approx(np.log(2))

    def test_string_labels(self, binary_cls):
        X, y = binary_cls
        labels = np.where(y == 1, "yes", "no")
        sl = SuperLearnerClassifier(_cls_bases()[:2], cv=4, random_state=0).fit(X, labels)
        assert set(sl.predict(X)) <= {"yes", "no"}
        assert list(sl.classes_) == ["no", "yes"]
