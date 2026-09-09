"""Tests for ROC AUC score."""
from __future__ import annotations

import numpy as np
import pytest

from ensemble_methods_kit import roc_auc_score


def test_perfect_separation_returns_one() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    assert roc_auc_score(y_true, y_score) == pytest.approx(1.0)


def test_random_classifier_returns_half() -> None:
    rng = np.random.default_rng(0)
    y_true = np.array([0, 1] * 50)
    y_score = rng.uniform(size=100)
    auc = roc_auc_score(y_true, y_score)
    assert 0.3 < auc < 0.7


def test_inverted_scores_returns_low() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.9, 0.8, 0.2, 0.1])
    assert roc_auc_score(y_true, y_score) == pytest.approx(0.0)


def test_mismatched_shapes_raise() -> None:
    with pytest.raises(ValueError, match="same shape"):
        roc_auc_score(np.array([0, 1]), np.array([0.1]))


def test_empty_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        roc_auc_score(np.array([]), np.array([]))


def test_single_class_raises() -> None:
    with pytest.raises(ValueError, match="exactly 2 classes"):
        roc_auc_score(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))


def test_all_positives_or_negatives_raises() -> None:
    with pytest.raises(ValueError, match="exactly 2 classes"):
        roc_auc_score(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3]))
    with pytest.raises(ValueError, match="exactly 2 classes"):
        roc_auc_score(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))


def test_ties_contribute_half() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.5, 0.5, 0.5, 0.5])
    assert roc_auc_score(y_true, y_score) == pytest.approx(0.5)
