"""Tests for error-correcting output codes."""

from __future__ import annotations

import numpy as np
import pytest

from ensemble_methods_kit import (
    DecisionTree,
    ECOCClassifier,
    LogisticRegression,
    accuracy_score,
    code_min_distance,
    make_code_matrix,
    train_test_split,
)


def _blobs(k=5, n=60, d=4, sep=3.0, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, sep, (k, d))
    X = np.vstack([rng.normal(c, 1.0, (n, d)) for c in centres])
    y = np.repeat(np.arange(k), n)
    return X, y


class TestCodeMatrix:
    def test_ovr_and_ovo_shapes(self):
        ovr = make_code_matrix(4, "ovr")
        assert ovr.shape == (4, 4) and np.all(np.diag(ovr) == 1)
        assert code_min_distance(ovr) == 2
        ovo = make_code_matrix(4, "ovo")
        assert ovo.shape == (4, 6)
        assert np.all((ovo != 0).sum(axis=0) == 2)
        assert code_min_distance(ovo) == pytest.approx(3.5)  # 1 conflict + 5 half-zeros

    def test_exhaustive_code_is_all_dichotomies(self):
        M = make_code_matrix(4, "exhaustive")
        assert M.shape == (4, 7)
        cols = {tuple(c) for c in M.T}
        assert len(cols) == 7
        # Dietterich & Bakiri: exhaustive codes have row distance 2^(K-2).
        assert code_min_distance(M) == 4
        with pytest.raises(ValueError):
            make_code_matrix(12, "exhaustive")

    @pytest.mark.parametrize("code", ["dense", "sparse"])
    def test_random_codes_are_valid(self, code):
        M = make_code_matrix(6, code, random_state=0)
        assert set(np.unique(M)) <= {-1.0, 0.0, 1.0}
        assert np.all((M > 0).any(axis=0) & (M < 0).any(axis=0))
        assert not np.any(np.all(M == 0, axis=1))
        assert code_min_distance(M) > 0
        signatures = set()
        for c in M.T:
            assert tuple(c) not in signatures and tuple(-c) not in signatures
            signatures.add(tuple(c))
        if code == "dense":
            assert set(np.unique(M)) == {-1.0, 1.0}

    def test_more_candidates_never_reduce_min_distance(self):
        d1 = code_min_distance(make_code_matrix(8, "dense", n_candidates=1, random_state=3))
        d50 = code_min_distance(make_code_matrix(8, "dense", n_candidates=50, random_state=3))
        assert d50 >= d1

    def test_errors(self):
        with pytest.raises(ValueError):
            make_code_matrix(1)
        with pytest.raises(ValueError):
            make_code_matrix(3, "nope")


class TestECOCClassifier:
    @pytest.mark.parametrize("code", ["ovr", "ovo", "exhaustive", "dense", "sparse"])
    @pytest.mark.parametrize("decoding", ["loss", "hamming"])
    def test_accuracy_on_blobs(self, code, decoding):
        X, y = _blobs(k=5)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
        clf = ECOCClassifier(code=code, decoding=decoding, random_state=0).fit(Xtr, ytr)
        assert accuracy_score(yte, clf.predict(Xte)) > 0.85
        assert len(clf.estimators_) == clf.code_matrix_.shape[1]

    def test_predict_proba_is_a_distribution_consistent_with_predict(self, multiclass_cls):
        X, y = multiclass_cls
        clf = ECOCClassifier(random_state=1).fit(X, y)
        P = clf.predict_proba(X)
        assert P.shape == (X.shape[0], 3)
        np.testing.assert_allclose(P.sum(axis=1), 1.0)
        np.testing.assert_array_equal(clf.classes_[P.argmax(axis=1)], clf.predict(X))

    def test_string_labels_and_logistic_base(self):
        X, y = _blobs(k=4, seed=2)
        labels = np.array(["ant", "bee", "cat", "dog"])[y]
        clf = ECOCClassifier(base_estimator=LogisticRegression(), code="dense", random_state=0)
        clf.fit(X, labels)
        assert set(clf.predict(X)) <= set(labels)
        assert clf.score(X, labels) > 0.85

    def test_error_correction_beats_ovr_when_learners_are_weak(self):
        X, y = _blobs(k=8, n=50, d=6, sep=1.6, seed=5)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)
        stump = DecisionTree(max_depth=2)
        ovr = ECOCClassifier(stump, code="ovr", decoding="hamming").fit(Xtr, ytr)
        dense = ECOCClassifier(stump, code="dense", code_size=4.0, decoding="loss", random_state=0).fit(Xtr, ytr)
        assert dense.score(Xte, yte) > ovr.score(Xte, yte)

    def test_custom_code_matrix_and_validation(self):
        X, y = _blobs(k=3)
        M = np.array([[1, 1, 0], [-1, 0, 1], [0, -1, -1]])
        clf = ECOCClassifier(code=M).fit(X, y)
        np.testing.assert_array_equal(clf.code_matrix_, M)
        assert clf.score(X, y) > 0.85
        with pytest.raises(ValueError):
            ECOCClassifier(code=np.ones((3, 2))).fit(X, y)
        with pytest.raises(ValueError):
            ECOCClassifier(code=np.array([[1, 2], [-1, 0], [0, -1]])).fit(X, y)
        with pytest.raises(ValueError):
            ECOCClassifier(code=M[:2]).fit(X, y)
        with pytest.raises(ValueError):
            ECOCClassifier(decoding="bad").fit(X, y)
        with pytest.raises(ValueError):
            ECOCClassifier().fit(X, np.zeros(len(y)))
        with pytest.raises(RuntimeError):
            ECOCClassifier().predict(X)

    def test_reproducible(self):
        X, y = _blobs(k=6, seed=7)
        a = ECOCClassifier(code="sparse", random_state=11).fit(X, y)
        b = ECOCClassifier(code="sparse", random_state=11).fit(X, y)
        np.testing.assert_array_equal(a.code_matrix_, b.code_matrix_)
        np.testing.assert_allclose(a.predict_proba(X), b.predict_proba(X))
