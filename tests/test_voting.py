"""Tests for the VotingClassifier."""

from __future__ import annotations

import numpy as np
import pytest

from ensemble_methods_kit import (
    DecisionTree,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
    accuracy_score,
)
from ensemble_methods_kit.utils import train_test_split


def _bases():
    return [
        ("rf", RandomForestClassifier(n_estimators=10, max_depth=5, random_state=1)),
        ("gbc", GradientBoostingClassifier(n_estimators=15, learning_rate=0.1,
                                           max_depth=2, random_state=1)),
        ("dt", DecisionTree(criterion="gini", max_depth=5, random_state=1)),
    ]


class _ConstantLabel:
    """Predicts one training label for every row."""

    def __init__(self, label):
        self.label = label

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self._n_features = np.asarray(X).shape[1]
        return self

    def predict(self, X):
        X = np.asarray(X)
        n = 1 if X.ndim == 1 else X.shape[0]
        return np.full(n, self.label, dtype=self.classes_.dtype)

    def predict_proba(self, X):
        pred = self.predict(X)
        proba = np.zeros((pred.shape[0], self.classes_.shape[0]), dtype=np.float64)
        index = {label: i for i, label in enumerate(self.classes_.tolist())}
        for row, label in enumerate(pred.tolist()):
            proba[row, index[label]] = 1.0
        return proba


class _ReversedProbaTree:
    """Decision tree whose ``classes_`` and probability columns are reversed."""

    def __init__(self, max_depth=4, random_state=0):
        self.max_depth = max_depth
        self.random_state = random_state

    def fit(self, X, y):
        self._tree = DecisionTree(max_depth=self.max_depth, random_state=self.random_state)
        self._tree.fit(X, y)
        self.classes_ = self._tree.classes_[::-1].copy()
        return self

    def predict_proba(self, X):
        return np.asarray(self._tree.predict_proba(X), dtype=np.float64)[:, ::-1]

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class _WeightedStub:
    def __init__(self):
        self.seen_weight = None

    def fit(self, X, y, sample_weight=None):
        self.seen_weight = None if sample_weight is None else np.asarray(sample_weight)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        n = 1 if X.ndim == 1 else X.shape[0]
        return np.full(n, self.classes_[0])

    def predict_proba(self, X):
        pred_n = self.predict(X).shape[0]
        proba = np.zeros((pred_n, self.classes_.shape[0]))
        proba[:, 0] = 1.0
        return proba


def test_voting_soft(binary_cls):
    X, y = binary_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    vc = VotingClassifier(_bases(), voting="soft").fit(Xtr, ytr)
    proba = vc.predict_proba(Xte)
    assert proba.shape == (len(yte), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert accuracy_score(yte, vc.predict(Xte)) > 0.9


def test_voting_hard(binary_cls):
    X, y = binary_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    vc = VotingClassifier(_bases(), voting="hard", weights=[2, 1, 1]).fit(Xtr, ytr)
    preds = vc.predict(Xte)
    assert set(np.unique(preds)).issubset({0, 1})
    assert accuracy_score(yte, preds) > 0.85


def test_voting_invalid_raises():
    with pytest.raises(ValueError):
        VotingClassifier([], voting="bad")


def test_voting_multiclass_soft_and_hard(multiclass_cls):
    X, y = multiclass_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    for voting in ("soft", "hard"):
        vc = VotingClassifier(_bases(), voting=voting).fit(Xtr, ytr)
        assert vc.classes_.tolist() == [0, 1, 2]
        preds = vc.predict(Xte)
        assert accuracy_score(yte, preds) > 0.9
        if voting == "soft":
            proba = vc.predict_proba(Xte)
            assert proba.shape == (len(yte), 3)
            assert np.allclose(proba.sum(axis=1), 1.0)
        else:
            with pytest.raises(AttributeError):
                vc.predict_proba(Xte)


def test_fit_clones_estimators(binary_cls):
    X, y = binary_cls
    tree = DecisionTree(max_depth=3, random_state=0)
    other = DecisionTree(max_depth=3, random_state=1)
    left = VotingClassifier([("t", tree)], voting="hard").fit(X, y)
    right = VotingClassifier([("t", tree), ("o", other)], voting="hard").fit(X, y)
    assert tree.classes_ is None
    assert other.classes_ is None
    assert left.named_estimators_["t"] is not tree
    assert left.estimators_[0] is not right.estimators_[0]
    # Refitting one ensemble must not change the other's fitted clone.
    before = left.predict(X).copy()
    right.fit(X, y)
    assert np.array_equal(before, left.predict(X))


def test_soft_vote_aligns_reversed_class_order(binary_cls):
    X, y = binary_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    vc = VotingClassifier(
        [
            ("forward", DecisionTree(max_depth=4, random_state=0)),
            ("reversed", _ReversedProbaTree(max_depth=4, random_state=0)),
        ],
        voting="soft",
    ).fit(Xtr, ytr)
    assert accuracy_score(yte, vc.predict(Xte)) > 0.9
    assert np.allclose(vc.predict_proba(Xte).sum(axis=1), 1.0)


def test_hard_vote_weights_and_ties():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 1, 0, 1])
    tied = VotingClassifier(
        [("a", _ConstantLabel(0)), ("b", _ConstantLabel(1))],
        voting="hard",
    ).fit(X, y)
    assert np.all(tied.predict(X) == 0)

    weighted = VotingClassifier(
        [("a", _ConstantLabel(0)), ("b", _ConstantLabel(1))],
        voting="hard",
        weights=[1.0, 3.0],
    ).fit(X, y)
    assert np.all(weighted.predict(X) == 1)


def test_soft_vote_weights_change_the_average():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])
    vc = VotingClassifier(
        [("zero", _ConstantLabel(0)), ("one", _ConstantLabel(1))],
        voting="soft",
        weights=[1.0, 3.0],
    ).fit(X, y)
    proba = vc.predict_proba(X)
    assert np.allclose(proba, [0.25, 0.75])
    assert np.all(vc.predict(X) == 1)


def test_string_and_noncontiguous_labels():
    X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array(["no", "no", "no", "yes", "yes", "yes"])
    vc = VotingClassifier(
        [("a", _ConstantLabel("yes")), ("b", _ConstantLabel("yes"))],
        voting="hard",
    ).fit(X, y)
    assert vc.predict(np.array([[0.0], [1.0]])).tolist() == ["yes", "yes"]

    y_nc = np.array([10, 10, 10, -2, -2, -2])
    soft = VotingClassifier(
        [("a", _ConstantLabel(-2)), ("b", _ConstantLabel(10))],
        voting="soft",
        weights=[1.0, 0.0],
    ).fit(X, y_nc)
    assert np.all(soft.predict(X) == -2)
    # classes_ is sorted: [-2, 10]. Weight sits entirely on the first estimator.
    assert np.allclose(soft.predict_proba(X)[:, 0], 1.0)


def test_drop_estimator_is_skipped():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 1, 0, 1])
    vc = VotingClassifier(
        [("keep", _ConstantLabel(1)), ("gone", "drop")],
        voting="hard",
        weights=[1.0, 50.0],
    ).fit(X, y)
    assert vc.named_estimators_["gone"] == "drop"
    assert len(vc.estimators_) == 1
    assert np.all(vc.predict(X) == 1)


def test_sample_weight_forwarded_or_rejected():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 1, 0, 1])
    weights = np.array([1.0, 1.0, 2.0, 2.0])
    stub = _WeightedStub()
    vc = VotingClassifier([("s", stub)], voting="soft").fit(X, y, sample_weight=weights)
    assert np.allclose(vc.named_estimators_["s"].seen_weight, weights)

    with pytest.raises(TypeError, match="sample weights"):
        VotingClassifier([("t", DecisionTree(max_depth=1))], voting="hard").fit(
            X, y, sample_weight=weights
        )


def test_transform_and_feature_names(binary_cls):
    X, y = binary_cls
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.3, random_state=42)
    soft = VotingClassifier(_bases(), voting="soft", flatten_transform=True).fit(Xtr, ytr)
    flat = soft.transform(Xtr[:4])
    assert flat.shape == (4, 3 * 2)
    assert soft.get_feature_names_out().shape == (6,)

    unflat = VotingClassifier(_bases(), voting="soft", flatten_transform=False).fit(Xtr, ytr)
    assert unflat.transform(Xtr[:4]).shape == (3, 4, 2)
    with pytest.raises(ValueError):
        unflat.get_feature_names_out()

    hard = VotingClassifier(_bases(), voting="hard").fit(Xtr, ytr)
    labels = hard.transform(Xtr[:4])
    assert labels.shape == (4, 3)
    assert hard.get_feature_names_out().tolist() == [
        "votingclassifier_rf",
        "votingclassifier_gbc",
        "votingclassifier_dt",
    ]


def test_feature_count_mismatch(binary_cls):
    X, y = binary_cls
    vc = VotingClassifier(
        [("t", DecisionTree(max_depth=2, random_state=0))],
        voting="hard",
    ).fit(X, y)
    with pytest.raises(ValueError, match="features"):
        vc.predict(X[:, :1])
    with pytest.raises(ValueError, match="features"):
        vc.transform(np.ones((2, X.shape[1] + 1)))


def test_predict_before_fit_and_bad_weights():
    X = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(RuntimeError, match="not fitted"):
        VotingClassifier(_bases(), voting="hard").predict(X)
    with pytest.raises(ValueError, match="weights"):
        VotingClassifier(_bases(), weights=[1.0, 2.0])
    with pytest.raises(ValueError, match="unique"):
        VotingClassifier([("a", _ConstantLabel(0)), ("a", _ConstantLabel(1))])

    class _LabelsOnly:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(np.asarray(X).shape[0])

    with pytest.raises(ValueError, match="predict_proba"):
        VotingClassifier([("a", _LabelsOnly())], voting="soft")
