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
