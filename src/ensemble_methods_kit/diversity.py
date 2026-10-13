"""Ensemble diversity measures and diversity-aware ensemble pruning.

Implements the oracle-output diversity measures surveyed by Kuncheva &
Whitaker (2003), "Measures of Diversity in Classifier Ensembles and Their
Relationship with the Ensemble Accuracy".  Every measure works on the
*oracle* matrix: ``oracle[i, j]`` is ``True`` when member ``i`` classifies
sample ``j`` correctly.

Pairwise measures (computed from the 2x2 contingency counts ``N11`` both
correct, ``N00`` both wrong, ``N10`` / ``N01`` exactly one correct):

* ``q_statistic``          Yule's Q, in [-1, 1]; lower = more diverse.
* ``correlation_coefficient``  rho, in [-1, 1]; lower = more diverse.
* ``disagreement``         (N01 + N10) / N; higher = more diverse.
* ``double_fault``         N00 / N; lower = more diverse.

Non-pairwise measures over the whole ensemble:

* ``entropy_diversity``        Kuncheva's E, in [0, 1]; higher = more diverse.
* ``kohavi_wolpert_variance``  KW, equals ``(L - 1) / (2L)`` times the
  averaged pairwise disagreement.
* ``interrater_agreement``     Fleiss-style kappa; lower = more diverse.

``diversity_prune`` greedily builds a sub-ensemble that trades individual
accuracy against disagreement with the members already selected (in the
spirit of Margineantu & Dietterich (1997) kappa pruning), and
``ensemble_member_predictions`` collects member predictions from any fitted
ensemble that exposes ``estimators_``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "oracle_matrix",
    "q_statistic",
    "correlation_coefficient",
    "disagreement",
    "double_fault",
    "pairwise_diversity",
    "average_pairwise_diversity",
    "entropy_diversity",
    "kohavi_wolpert_variance",
    "interrater_agreement",
    "diversity_report",
    "ensemble_member_predictions",
    "majority_vote",
    "diversity_prune",
]

_PAIRWISE = ("q_statistic", "correlation", "disagreement", "double_fault")


def oracle_matrix(predictions, y) -> np.ndarray:
    """Return the boolean oracle matrix ``predictions == y``.

    ``predictions`` has shape ``(n_members, n_samples)`` (labels predicted by
    each member) and ``y`` shape ``(n_samples,)``.
    """
    P = np.asarray(predictions)
    y = np.asarray(y)
    if P.ndim != 2:
        raise ValueError("predictions must be 2D (n_members, n_samples)")
    if y.ndim != 1 or y.shape[0] != P.shape[1]:
        raise ValueError("y must be 1D with length equal to predictions.shape[1]")
    if P.shape[1] == 0:
        raise ValueError("need at least one sample")
    return P == y[None, :]


def _as_oracle_pair(a, b):
    a = np.asarray(a, dtype=bool).ravel()
    b = np.asarray(b, dtype=bool).ravel()
    if a.shape != b.shape:
        raise ValueError("oracle vectors must have the same length")
    if a.size == 0:
        raise ValueError("oracle vectors must be non-empty")
    return a, b


def _counts(a, b):
    a, b = _as_oracle_pair(a, b)
    n11 = float(np.sum(a & b))
    n00 = float(np.sum(~a & ~b))
    n10 = float(np.sum(a & ~b))
    n01 = float(np.sum(~a & b))
    return n11, n00, n10, n01


def q_statistic(correct_a, correct_b) -> float:
    """Yule's Q statistic between two oracle vectors.

    Returns 0.0 when the statistic is undefined (zero denominator), which
    happens e.g. when one member is always right.
    """
    n11, n00, n10, n01 = _counts(correct_a, correct_b)
    den = n11 * n00 + n01 * n10
    if den == 0:
        return 0.0
    return (n11 * n00 - n01 * n10) / den


def correlation_coefficient(correct_a, correct_b) -> float:
    """Pearson correlation (phi coefficient) between two oracle vectors."""
    n11, n00, n10, n01 = _counts(correct_a, correct_b)
    den = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    if den == 0:
        return 0.0
    return float((n11 * n00 - n01 * n10) / den)


def disagreement(correct_a, correct_b) -> float:
    """Fraction of samples where exactly one of the two members is correct."""
    n11, n00, n10, n01 = _counts(correct_a, correct_b)
    return (n01 + n10) / (n11 + n00 + n10 + n01)


def double_fault(correct_a, correct_b) -> float:
    """Fraction of samples misclassified by both members."""
    n11, n00, n10, n01 = _counts(correct_a, correct_b)
    return n00 / (n11 + n00 + n10 + n01)


_PAIR_FUNCS = {
    "q_statistic": q_statistic,
    "correlation": correlation_coefficient,
    "disagreement": disagreement,
    "double_fault": double_fault,
}


def _check_oracle(oracle) -> np.ndarray:
    O = np.asarray(oracle, dtype=bool)
    if O.ndim != 2:
        raise ValueError("oracle must be 2D (n_members, n_samples)")
    if O.shape[0] < 2:
        raise ValueError("need at least two ensemble members")
    if O.shape[1] == 0:
        raise ValueError("need at least one sample")
    return O


def pairwise_diversity(oracle, measure: str = "disagreement") -> np.ndarray:
    """Symmetric ``(L, L)`` matrix of a pairwise measure.

    The diagonal holds the measure of a member against itself (1 for Q and
    rho, 0 for disagreement, the error rate for double-fault).
    """
    if measure not in _PAIR_FUNCS:
        raise ValueError(f"measure must be one of {_PAIRWISE}")
    O = _check_oracle(oracle)
    func = _PAIR_FUNCS[measure]
    L = O.shape[0]
    M = np.zeros((L, L), dtype=np.float64)
    for i in range(L):
        for j in range(i, L):
            M[i, j] = M[j, i] = func(O[i], O[j])
    return M


def average_pairwise_diversity(oracle, measure: str = "disagreement") -> float:
    """Mean of a pairwise measure over all ``L (L - 1) / 2`` distinct pairs."""
    M = pairwise_diversity(oracle, measure)
    iu = np.triu_indices(M.shape[0], k=1)
    return float(np.mean(M[iu]))


def entropy_diversity(oracle) -> float:
    """Kuncheva's entropy measure E (0 = no diversity, 1 = maximal)."""
    O = _check_oracle(oracle)
    L = O.shape[0]
    correct = O.sum(axis=0)
    denom = L - int(np.ceil(L / 2.0))
    return float(np.mean(np.minimum(correct, L - correct)) / denom)


def kohavi_wolpert_variance(oracle) -> float:
    """Kohavi-Wolpert variance of the oracle outputs."""
    O = _check_oracle(oracle)
    L, N = O.shape
    correct = O.sum(axis=0).astype(np.float64)
    return float(np.sum(correct * (L - correct)) / (N * L * L))


def interrater_agreement(oracle) -> float:
    """Measurement of inter-rater agreement kappa (Fleiss 1981).

    Returns 1.0 when the members' average accuracy is 0 or 1 (no variance
    in the oracle outputs, hence no measurable disagreement).
    """
    O = _check_oracle(oracle)
    L, N = O.shape
    correct = O.sum(axis=0).astype(np.float64)
    p_bar = float(O.mean())
    den = N * (L - 1) * p_bar * (1.0 - p_bar)
    if den == 0:
        return 1.0
    return float(1.0 - np.sum(correct * (L - correct)) / L / den)


def diversity_report(predictions, y) -> Dict[str, object]:
    """Compute every measure plus per-member and majority-vote accuracy."""
    O = oracle_matrix(predictions, y)
    _check_oracle(O)
    return {
        "n_members": int(O.shape[0]),
        "n_samples": int(O.shape[1]),
        "member_accuracy": O.mean(axis=1).tolist(),
        "mean_member_accuracy": float(O.mean()),
        "majority_vote_accuracy": float(np.mean(majority_vote(predictions) == np.asarray(y))),
        "oracle_accuracy": float(np.mean(O.any(axis=0))),
        "q_statistic": average_pairwise_diversity(O, "q_statistic"),
        "correlation": average_pairwise_diversity(O, "correlation"),
        "disagreement": average_pairwise_diversity(O, "disagreement"),
        "double_fault": average_pairwise_diversity(O, "double_fault"),
        "entropy": entropy_diversity(O),
        "kohavi_wolpert": kohavi_wolpert_variance(O),
        "kappa": interrater_agreement(O),
    }


def ensemble_member_predictions(ensemble, X, estimators: Optional[Sequence] = None) -> np.ndarray:
    """Stack the label predictions of each fitted member: ``(L, n_samples)``.

    ``estimators`` defaults to ``ensemble.estimators_``.  Entries may be bare
    estimators or ``(name, estimator)`` pairs.  For ensembles trained on
    feature subsets (``estimators_features_``) or rotations
    (``rotation_matrices_``) the member's view of ``X`` is reconstructed; if
    the ensemble stores ``classes_`` and members predict encoded indices,
    the indices are mapped back to labels.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    members = list(estimators) if estimators is not None else list(getattr(ensemble, "estimators_", []) or [])
    if not members:
        raise ValueError("ensemble has no fitted members (estimators_ is empty)")
    feats = getattr(ensemble, "estimators_features_", None)
    rots = getattr(ensemble, "rotation_matrices_", None)
    classes = getattr(ensemble, "classes_", None)
    rows: List[np.ndarray] = []
    for i, member in enumerate(members):
        est = member[1] if isinstance(member, tuple) and len(member) == 2 else member
        Xi = X
        if feats is not None and estimators is None:
            Xi = X[:, np.asarray(feats[i])]
        elif rots is not None and estimators is None:
            Xi = X @ np.asarray(rots[i])
        pred = np.asarray(est.predict(Xi))
        if classes is not None:
            classes_arr = np.asarray(classes)
            if not np.all(np.isin(pred, classes_arr)) and np.issubdtype(pred.dtype, np.number):
                pred = classes_arr[pred.astype(int)]
        rows.append(pred)
    return np.vstack(rows)


def majority_vote(predictions, weights=None) -> np.ndarray:
    """Plurality vote over member predictions (ties -> smallest label)."""
    P = np.asarray(predictions)
    if P.ndim != 2 or P.shape[0] == 0:
        raise ValueError("predictions must be 2D (n_members, n_samples)")
    w = np.ones(P.shape[0]) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (P.shape[0],):
        raise ValueError("weights must have one entry per member")
    labels = np.unique(P)
    tallies = np.stack([((P == lab) * w[:, None]).sum(axis=0) for lab in labels])
    return labels[np.argmax(tallies, axis=0)]


def diversity_prune(
    predictions,
    y,
    n_select: int,
    alpha: float = 0.5,
    measure: str = "disagreement",
) -> Dict[str, object]:
    """Greedy diversity-aware selection of ``n_select`` ensemble members.

    Starts from the most accurate member, then repeatedly adds the member
    maximising ``alpha * accuracy + (1 - alpha) * diversity_to_selected``,
    where diversity is the mean pairwise ``measure`` to the members already
    chosen (``disagreement`` as-is; for ``q_statistic``, ``correlation`` and
    ``double_fault`` lower values mean more diverse, so ``1 - value`` is
    used, rescaled to [0, 1] for Q / rho).

    ``alpha=1`` reduces to picking the top-``n_select`` accurate members;
    ``alpha=0`` is pure diversity after the first pick.

    Returns a dict with ``selected`` (member indices in pick order),
    ``scores`` (selection score at each step), ``pruned_accuracy`` (majority
    vote of the selected members) and ``full_accuracy``.
    """
    if measure not in _PAIR_FUNCS:
        raise ValueError(f"measure must be one of {_PAIRWISE}")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    P = np.asarray(predictions)
    O = oracle_matrix(P, y)
    L = O.shape[0]
    if not isinstance(n_select, (int, np.integer)) or not 1 <= n_select <= L:
        raise ValueError(f"n_select must be an integer in [1, {L}]")
    acc = O.mean(axis=1)
    if L >= 2:
        M = pairwise_diversity(O, measure)
    else:
        M = np.zeros((1, 1))
    if measure == "disagreement":
        D = M
    elif measure == "double_fault":
        D = 1.0 - M
    else:  # q_statistic / correlation in [-1, 1]
        D = (1.0 - M) / 2.0

    first = int(np.argmax(acc))
    selected = [first]
    scores = [float(acc[first])]
    remaining = [i for i in range(L) if i != first]
    while len(selected) < n_select:
        best, best_score = None, -np.inf
        for i in remaining:
            div = float(np.mean(D[i, selected]))
            s = alpha * acc[i] + (1.0 - alpha) * div
            if s > best_score + 1e-12:
                best, best_score = i, s
        selected.append(best)
        scores.append(float(best_score))
        remaining.remove(best)

    y = np.asarray(y)
    return {
        "selected": selected,
        "scores": scores,
        "member_accuracy": acc.tolist(),
        "pruned_accuracy": float(np.mean(majority_vote(P[selected]) == y)),
        "full_accuracy": float(np.mean(majority_vote(P) == y)),
    }
