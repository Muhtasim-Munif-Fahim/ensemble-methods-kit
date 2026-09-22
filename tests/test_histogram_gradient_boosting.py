"""Tests for HistogramGradientBoostingClassifier."""

from __future__ import annotations

import numpy as np
import pytest

from ensemble_methods_kit import (
    HistogramGradientBoostingClassifier,
    accuracy_score,
    log_loss,
)
from ensemble_methods_kit.utils import train_test_split


def _fit_binary(X, y, **kwargs):
    params = dict(n_estimators=40, learning_rate=0.1, max_depth=3, random_state=0)
    params.update(kwargs)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = HistogramGradientBoostingClassifier(**params).fit(Xtr, ytr)
    return clf, Xte, yte


def test_histogram_gradient_boosting_binary_proba(binary_cls):
    X, y = binary_cls
    clf, Xte, yte = _fit_binary(X, y)
    assert accuracy_score(yte, clf.predict(Xte)) > 0.9
    proba = clf.predict_proba(Xte)
    assert proba.shape == (len(yte), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all(proba >= 0.0)
    assert log_loss(yte, proba) < 0.4
    assert clf.n_iter_ == 40
    assert len(clf.estimators_) == 40
    assert all(len(stage) == 1 for stage in clf.estimators_)
    assert len(clf.train_score_) == 40
    assert clf.train_score_[-1] <= clf.train_score_[0]


def test_histogram_gradient_boosting_multiclass(multiclass_cls):
    X, y = multiclass_cls
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    clf = HistogramGradientBoostingClassifier(
        n_estimators=30, learning_rate=0.1, max_depth=3, random_state=1
    ).fit(Xtr, ytr)
    assert clf.classes_.tolist() == [0, 1, 2]
    assert clf.n_classes_ == 3
    assert accuracy_score(yte, clf.predict(Xte)) > 0.85
    proba = clf.predict_proba(Xte)
    assert proba.shape == (len(yte), 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert all(len(stage) == 3 for stage in clf.estimators_)


def test_staged_predict_proba_matches_final(binary_cls):
    X, y = binary_cls
    clf, Xte, yte = _fit_binary(X, y, n_estimators=15, max_depth=2)
    stages = list(clf.staged_predict_proba(Xte))
    assert len(stages) == clf.n_estimators
    for proba in stages:
        assert proba.shape == (len(yte), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.allclose(stages[-1], clf.predict_proba(Xte))
    labels = list(clf.staged_predict(Xte))
    assert len(labels) == clf.n_estimators
    assert np.array_equal(labels[-1], clf.predict(Xte))
    assert log_loss(yte, stages[0]) >= log_loss(yte, stages[-1]) - 1e-9


def test_more_estimators_reduces_loss(binary_cls):
    X, y = binary_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    small = HistogramGradientBoostingClassifier(
        n_estimators=5, learning_rate=0.1, max_depth=2, random_state=0
    ).fit(Xtr, ytr)
    large = HistogramGradientBoostingClassifier(
        n_estimators=30, learning_rate=0.1, max_depth=2, random_state=0
    ).fit(Xtr, ytr)
    assert log_loss(yte, small.predict_proba(Xte)) >= log_loss(
        yte, large.predict_proba(Xte)
    )


def test_reproducible_with_subsampling(binary_cls):
    X, y = binary_cls
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)
    kwargs = dict(
        n_estimators=15, learning_rate=0.1, max_depth=2, subsample=0.7, random_state=5
    )
    a = HistogramGradientBoostingClassifier(**kwargs).fit(Xtr, ytr).predict(Xte)
    b = HistogramGradientBoostingClassifier(**kwargs).fit(Xtr, ytr).predict(Xte)
    assert np.array_equal(a, b)
    assert accuracy_score(yte, a) > 0.85


def test_noncontiguous_and_string_labels(binary_cls):
    X, y = binary_cls
    y_int = np.where(y == 0, 2, 5)
    clf, Xte, yte = _fit_binary(X, y_int, n_estimators=20)
    assert clf.classes_.tolist() == [2, 5]
    assert set(np.unique(clf.predict(Xte))).issubset({2, 5})
    assert accuracy_score(yte, clf.predict(Xte)) > 0.9

    y_str = np.where(y == 0, "no", "yes")
    clf_str, Xte_str, yte_str = _fit_binary(X, y_str, n_estimators=20)
    assert clf_str.classes_.tolist() == ["no", "yes"]
    assert set(np.unique(clf_str.predict(Xte_str))).issubset({"no", "yes"})
    assert accuracy_score(yte_str, clf_str.predict(Xte_str)) > 0.9


def test_coarse_bins_still_separate(binary_cls):
    X, y = binary_cls
    clf, Xte, yte = _fit_binary(X, y, n_estimators=20, max_bins=2, max_depth=2)
    assert accuracy_score(yte, clf.predict(Xte)) > 0.9
    assert all(edges.size <= 1 for edges in clf.bin_thresholds_)


def test_constant_feature_is_not_split():
    rng = np.random.default_rng(0)
    signal = rng.normal(size=80)
    X = np.column_stack([signal, np.ones(80)])
    y = (signal > 0).astype(int)
    clf = HistogramGradientBoostingClassifier(
        n_estimators=10, max_depth=2, random_state=0
    ).fit(X, y)
    assert clf.bin_thresholds_[1].size == 0
    assert clf.bin_thresholds_[0].size >= 1
    importances = clf.feature_importances_
    assert importances.shape == (2,)
    assert importances[1] == 0.0
    assert importances[0] == pytest.approx(1.0)


def test_feature_importances_rank_signal():
    rng = np.random.default_rng(0)
    signal = rng.normal(size=240)
    X = np.column_stack(
        [signal, rng.normal(size=240), rng.normal(size=240), np.zeros(240)]
    )
    y = (signal > 0).astype(int)
    clf = HistogramGradientBoostingClassifier(
        n_estimators=20, max_depth=3, random_state=0
    ).fit(X, y)
    importances = clf.feature_importances_
    assert importances.shape == (4,)
    assert importances.min() >= 0.0
    assert importances.sum() == pytest.approx(1.0)
    assert importances[0] == importances.max()
    assert importances[3] == 0.0


def test_quantile_bin_count_respects_max_bins():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 1))
    y = (X[:, 0] > 0).astype(int)
    clf = HistogramGradientBoostingClassifier(
        n_estimators=2, max_bins=4, random_state=0
    ).fit(X, y)
    assert 1 <= clf.bin_thresholds_[0].size <= 3


def test_parameter_validation():
    with pytest.raises(ValueError, match="n_estimators"):
        HistogramGradientBoostingClassifier(n_estimators=0)
    with pytest.raises(ValueError, match="learning_rate"):
        HistogramGradientBoostingClassifier(learning_rate=0.0)
    with pytest.raises(ValueError, match="max_depth"):
        HistogramGradientBoostingClassifier(max_depth=0)
    with pytest.raises(ValueError, match="max_bins"):
        HistogramGradientBoostingClassifier(max_bins=1)
    with pytest.raises(ValueError, match="min_samples_leaf"):
        HistogramGradientBoostingClassifier(min_samples_leaf=0)
    with pytest.raises(ValueError, match="l2_regularization"):
        HistogramGradientBoostingClassifier(l2_regularization=-0.1)
    with pytest.raises(ValueError, match="subsample"):
        HistogramGradientBoostingClassifier(subsample=0.0)
    with pytest.raises(ValueError, match="subsample"):
        HistogramGradientBoostingClassifier(subsample=1.5)


def test_predict_before_fit_raises():
    clf = HistogramGradientBoostingClassifier(n_estimators=2)
    X = np.array([[0.0, 1.0]])
    with pytest.raises(RuntimeError, match="not fitted"):
        clf.predict(X)
    with pytest.raises(RuntimeError, match="not fitted"):
        clf.predict_proba(X)
    with pytest.raises(RuntimeError, match="not fitted"):
        list(clf.staged_predict_proba(X))
    with pytest.raises(RuntimeError, match="not fitted"):
        clf.feature_importances_


def test_single_class_and_shape_errors():
    clf = HistogramGradientBoostingClassifier(n_estimators=2)
    with pytest.raises(ValueError, match="at least 2 classes"):
        clf.fit(np.array([[0.0], [1.0]]), np.array([1, 1]))
    with pytest.raises(ValueError, match="same number of samples"):
        clf.fit(np.array([[0.0], [1.0]]), np.array([0, 1, 0]))
    with pytest.raises(ValueError, match="finite"):
        clf.fit(np.array([[0.0], [np.nan]]), np.array([0, 1]))


def test_feature_count_mismatch(binary_cls):
    X, y = binary_cls
    clf = HistogramGradientBoostingClassifier(n_estimators=3, random_state=0).fit(X, y)
    with pytest.raises(ValueError, match="features"):
        clf.predict(X[:, :2])


def test_refit_resets_iterations(binary_cls):
    X, y = binary_cls
    clf = HistogramGradientBoostingClassifier(n_estimators=4, random_state=0)
    clf.fit(X, y)
    assert clf.n_iter_ == 4
    clf.n_estimators = 7
    clf.fit(X, y)
    assert clf.n_iter_ == 7
    assert len(clf.train_score_) == 7
    assert len(clf.estimators_) == 7
