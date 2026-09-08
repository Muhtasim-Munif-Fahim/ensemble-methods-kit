"""Tests for classification metric functions."""
from __future__ import annotations

import numpy as np

from ensemble_methods_kit import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def test_confusion_matrix_binary() -> None:
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    cm = confusion_matrix(y_true, y_pred)
    np.testing.assert_array_equal(cm, [[1, 1], [0, 2]])


def test_confusion_matrix_all_correct() -> None:
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    cm = confusion_matrix(y_true, y_pred)
    np.testing.assert_array_equal(cm, [[2, 0], [0, 2]])


def test_accuracy_score() -> None:
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 0]
    assert accuracy_score(y_true, y_pred) == 0.75


def test_precision_score() -> None:
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    # TP=2, FP=1 -> precision = 2/3
    assert abs(precision_score(y_true, y_pred) - 2 / 3) < 1e-9


def test_precision_score_zero_division() -> None:
    y_true = [0, 0]
    y_pred = [1, 1]
    # TP=0, FP=2 -> precision = 0
    assert precision_score(y_true, y_pred) == 0.0


def test_recall_score() -> None:
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    # TP=2, FN=0 -> recall = 1.0
    assert recall_score(y_true, y_pred) == 1.0


def test_recall_score_partial() -> None:
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 0]
    # TP=1, FN=1 -> recall = 0.5
    assert recall_score(y_true, y_pred) == 0.5


def test_f1_score() -> None:
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    p = 2 / 3
    r = 1.0
    expected = 2 * p * r / (p + r)
    assert abs(f1_score(y_true, y_pred) - expected) < 1e-9


def test_f1_score_perfect() -> None:
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    assert f1_score(y_true, y_pred) == 1.0


def test_f1_score_zero_division() -> None:
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 0, 0]
    # No positive predictions at all
    assert f1_score(y_true, y_pred) == 0.0


def test_metrics_with_numpy_arrays() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0, 1])
    acc = accuracy_score(y_true, y_pred)
    assert acc == 4 / 6
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    assert p > 0
    assert r > 0
    assert f1 > 0
