"""Tests for ensemble diversity measures and diversity-based pruning."""

from __future__ import annotations

import numpy as np
import pytest

from ensemble_methods_kit import (
    BaggingClassifier,
    RandomForestClassifier,
    RandomSubspaceClassifier,
    average_pairwise_diversity,
    correlation_coefficient,
    disagreement,
    diversity_prune,
    diversity_report,
    double_fault,
    ensemble_member_predictions,
    entropy_diversity,
    interrater_agreement,
    kohavi_wolpert_variance,
    majority_vote,
    oracle_matrix,
    pairwise_diversity,
    q_statistic,
    train_test_split,
)


def test_pairwise_measures_hand_counts():
    # N11=3, N00=1, N10=2, N01=4 -> N=10
    a = np.array([1, 1, 1, 0, 1, 1, 0, 0, 0, 0], dtype=bool)
    b = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 1], dtype=bool)
    assert q_statistic(a, b) == pytest.approx((3 * 1 - 4 * 2) / (3 * 1 + 4 * 2))
    rho = (3 * 1 - 4 * 2) / np.sqrt((3 + 2) * (4 + 1) * (3 + 4) * (2 + 1))
    assert correlation_coefficient(a, b) == pytest.approx(rho)
    assert disagreement(a, b) == pytest.approx(0.6)
    assert double_fault(a, b) == pytest.approx(0.1)


def test_correlation_matches_numpy_corrcoef():
    rng = np.random.default_rng(0)
    a = rng.random(200) < 0.7
    b = rng.random(200) < 0.6
    expected = np.corrcoef(a.astype(float), b.astype(float))[0, 1]
    assert correlation_coefficient(a, b) == pytest.approx(expected)


def test_identical_members_have_no_diversity():
    o = np.array([[1, 0, 1, 1, 0]] * 3, dtype=bool)
    assert average_pairwise_diversity(o, "disagreement") == 0.0
    assert average_pairwise_diversity(o, "q_statistic") == pytest.approx(1.0)
    assert entropy_diversity(o) == 0.0
    assert kohavi_wolpert_variance(o) == 0.0
    assert interrater_agreement(o) == pytest.approx(1.0)


def test_undefined_q_returns_zero():
    a = np.ones(5, dtype=bool)
    b = np.array([1, 0, 1, 0, 1], dtype=bool)
    assert q_statistic(a, b) == 0.0
    assert correlation_coefficient(a, b) == 0.0


def test_kw_equals_scaled_disagreement():
    rng = np.random.default_rng(1)
    o = rng.random((7, 300)) < 0.65
    L = o.shape[0]
    dis = average_pairwise_diversity(o, "disagreement")
    assert kohavi_wolpert_variance(o) == pytest.approx((L - 1) / (2 * L) * dis)


def test_kappa_relation_to_kw():
    rng = np.random.default_rng(2)
    o = rng.random((5, 400)) < 0.7
    L = o.shape[0]
    p = o.mean()
    kw = kohavi_wolpert_variance(o)
    assert interrater_agreement(o) == pytest.approx(1 - L / ((L - 1) * p * (1 - p)) * kw)


def test_entropy_max_when_split_evenly():
    # 4 members, every sample has exactly 2 correct -> E = 1
    o = np.array(
        [[1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1]], dtype=bool
    )
    assert entropy_diversity(o) == pytest.approx(1.0)


def test_pairwise_matrix_symmetric_and_diag():
    rng = np.random.default_rng(3)
    o = rng.random((4, 50)) < 0.6
    M = pairwise_diversity(o, "double_fault")
    assert np.allclose(M, M.T)
    assert np.allclose(np.diag(M), 1 - o.mean(axis=1))
    D = pairwise_diversity(o, "disagreement")
    assert np.allclose(np.diag(D), 0.0)


def test_validation_errors():
    with pytest.raises(ValueError):
        pairwise_diversity(np.ones((1, 4), dtype=bool))
    with pytest.raises(ValueError):
        pairwise_diversity(np.ones((3, 4), dtype=bool), measure="nope")
    with pytest.raises(ValueError):
        oracle_matrix(np.zeros((2, 3)), np.zeros(4))
    with pytest.raises(ValueError):
        disagreement(np.ones(3, dtype=bool), np.ones(4, dtype=bool))
    P = np.zeros((3, 5), dtype=int)
    with pytest.raises(ValueError):
        diversity_prune(P, np.zeros(5), n_select=4)
    with pytest.raises(ValueError):
        diversity_prune(P, np.zeros(5), n_select=2, alpha=1.5)


def test_majority_vote_and_weights():
    P = np.array([[0, 1, 2], [0, 2, 2], [1, 1, 0]])
    assert majority_vote(P).tolist() == [0, 1, 2]
    assert majority_vote(P, weights=[0.1, 0.1, 5.0]).tolist() == [1, 1, 0]


def test_diversity_prune_alpha_one_is_top_accuracy():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 200)
    accs = [0.9, 0.6, 0.8, 0.7, 0.85]
    P = np.array([np.where(rng.random(200) < a, y, 1 - y) for a in accs])
    out = diversity_prune(P, y, n_select=3, alpha=1.0)
    member_acc = np.array(out["member_accuracy"])
    assert sorted(out["selected"]) == sorted(np.argsort(-member_acc)[:3].tolist())


def test_diversity_prune_prefers_diverse_over_duplicates():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 300)
    base = np.where(rng.random(300) < 0.8, y, 1 - y)
    other1 = np.where(rng.random(300) < 0.75, y, 1 - y)
    other2 = np.where(rng.random(300) < 0.75, y, 1 - y)
    P = np.vstack([base, base, base, other1, other2])
    out = diversity_prune(P, y, n_select=3, alpha=0.3)
    assert out["selected"][0] in (0, 1, 2)
    assert set(out["selected"][1:]) == {3, 4}
    assert out["pruned_accuracy"] >= out["member_accuracy"][0] - 1e-12
    # all other measures work too
    for m in ("q_statistic", "correlation", "double_fault"):
        res = diversity_prune(P, y, n_select=3, alpha=0.3, measure=m)
        assert len(set(res["selected"])) == 3


def test_ensemble_member_predictions_and_report_on_real_ensembles():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(240, 5))
    y = np.where(X[:, 0] + 0.5 * X[:, 1] - X[:, 2] > 0, "pos", "neg")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
    for model in (
        BaggingClassifier(n_estimators=7, random_state=0),
        RandomForestClassifier(n_estimators=7, random_state=0),
        RandomSubspaceClassifier(n_estimators=7, max_features=0.6, random_state=0),
    ):
        model.fit(Xtr, ytr)
        P = ensemble_member_predictions(model, Xte)
        assert P.shape == (7, len(yte))
        assert set(np.unique(P)) <= {"pos", "neg"}
        rep = diversity_report(P, yte)
        assert rep["n_members"] == 7
        assert 0.0 <= rep["disagreement"] <= 1.0
        assert 0.0 <= rep["entropy"] <= 1.0
        assert rep["oracle_accuracy"] >= max(rep["member_accuracy"])
        pruned = diversity_prune(P, yte, n_select=3)
        assert len(pruned["selected"]) == 3


def test_ensemble_member_predictions_requires_members():
    class Empty:
        estimators_: list = []

    with pytest.raises(ValueError):
        ensemble_member_predictions(Empty(), np.zeros((2, 2)))
