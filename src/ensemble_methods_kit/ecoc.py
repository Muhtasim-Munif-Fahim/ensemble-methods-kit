"""Error-Correcting Output Codes (ECOC) for multiclass classification.

Dietterich & Bakiri (1995), *Solving multiclass learning problems via
error-correcting output codes*, reduce a ``K``-class problem to ``L`` binary
problems. Each class ``k`` gets a codeword ``M[k] in {-1, 0, +1}^L``; the
binary learner for column ``l`` separates the classes coded ``+1`` from those
coded ``-1``, and classes coded ``0`` (Allwein, Schapire & Singer, 2000) are
left out of that learner's training set. A test point is assigned to the
class whose codeword is closest to the vector of learner outputs, so a few
wrong dichotomies can be corrected when codewords are far apart.

Code designs (``code``):

* ``"ovr"`` - one-vs-rest identity code (``L = K``).
* ``"ovo"`` - one-vs-one ternary code (``L = K(K-1)/2``).
* ``"exhaustive"`` - all ``2^(K-1) - 1`` distinct dichotomies (``K <= 11``).
* ``"dense"`` - random ``{-1, +1}`` code, ``L = ceil(code_size * K)`` columns
  (default ``code_size = 10 log2 K / K`` per Allwein et al.).
* ``"sparse"`` - random ternary code with ``P(0) = 1/2``.
* an explicit ``(K, L)`` matrix.

For random codes ``n_candidates`` matrices are drawn and the one with the
largest minimum row (Hamming) distance is kept; columns that are constant or
duplicate (up to sign) are rejected.

Decoding (``decoding``):

* ``"hamming"`` - generalised Hamming distance
  ``sum_l (1 - sign(f_l) M_kl) / 2`` on hard outputs (0 entries cost 1/2);
  ties are broken by the soft margins.
* ``"loss"`` - Allwein et al. loss-based decoding
  ``sum_l L(M_kl f_l)`` with the exponential loss ``exp(-z)`` on margins
  ``f_l = 2 p_l - 1`` from each learner's ``predict_proba``. Zero entries
  contribute the constant ``L(0)``.

``predict_proba`` turns negative decoding losses into a softmax over classes.
"""

from __future__ import annotations

import itertools
import math
from typing import Optional, Union

import numpy as np

from .utils import DecisionTree, clone_estimator

__all__ = ["ECOCClassifier", "make_code_matrix", "code_min_distance"]


def code_min_distance(M) -> float:
    """Minimum generalised Hamming distance between distinct codewords."""
    M = np.asarray(M, dtype=float)
    K = M.shape[0]
    best = math.inf
    for a in range(K):
        for b in range(a + 1, K):
            d = float(np.sum((1.0 - M[a] * M[b]) / 2.0))
            best = min(best, d)
    return best


def _valid_columns(M: np.ndarray) -> np.ndarray:
    """Keep columns with both signs, dropping duplicates/complements."""
    keep = []
    seen = set()
    for l in range(M.shape[1]):
        col = M[:, l]
        if not (np.any(col > 0) and np.any(col < 0)):
            continue
        key = tuple(col.astype(int))
        neg = tuple((-col).astype(int))
        if key in seen or neg in seen:
            continue
        seen.add(key)
        keep.append(l)
    return M[:, keep]


def make_code_matrix(
    n_classes: int,
    code: str = "dense",
    code_size: Optional[float] = None,
    n_candidates: int = 100,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Build a ``(n_classes, L)`` ECOC code matrix with entries in {-1, 0, 1}."""
    K = int(n_classes)
    if K < 2:
        raise ValueError("ECOC needs at least two classes")
    if code == "ovr":
        M = -np.ones((K, K))
        np.fill_diagonal(M, 1.0)
        return M
    if code == "ovo":
        pairs = list(itertools.combinations(range(K), 2))
        M = np.zeros((K, len(pairs)))
        for l, (a, b) in enumerate(pairs):
            M[a, l], M[b, l] = 1.0, -1.0
        return M
    if code == "exhaustive":
        if K > 11:
            raise ValueError("exhaustive codes are limited to <= 11 classes")
        # Class 0 is fixed to +1; the other K-1 bits enumerate all
        # dichotomies except the constant one.
        cols = []
        for mask in range(2 ** (K - 1) - 1):
            col = np.ones(K)
            for k in range(1, K):
                if not (mask >> (k - 1)) & 1:
                    col[k] = -1.0
            cols.append(col)
        return np.array(cols).T
    if code not in ("dense", "sparse"):
        raise ValueError(
            "code must be 'ovr', 'ovo', 'exhaustive', 'dense', 'sparse' or a matrix"
        )
    if code_size is None:
        code_size = max(10.0 * math.log2(K) / K, 1.0) if code == "dense" else max(15.0 * math.log2(K) / K, 1.0)
    if code_size <= 0:
        raise ValueError("code_size must be positive")
    L = max(int(math.ceil(code_size * K)), 1)
    rng = np.random.default_rng(random_state)
    best, best_d = None, -1.0
    for _ in range(max(int(n_candidates), 1)):
        if code == "dense":
            M = rng.choice([-1.0, 1.0], size=(K, L))
        else:
            M = rng.choice([-1.0, 0.0, 1.0], size=(K, L), p=[0.25, 0.5, 0.25])
        M = _valid_columns(M)
        if M.shape[1] == 0:
            continue
        d = code_min_distance(M)
        # Every class must take part in at least one dichotomy.
        if np.any(np.all(M == 0, axis=1)):
            continue
        if d > best_d:
            best, best_d = M, d
    if best is None:
        raise RuntimeError("could not draw a valid random code; increase code_size")
    return best


class ECOCClassifier:
    """Multiclass classifier from binary learners via error-correcting codes.

    Parameters
    ----------
    base_estimator :
        Binary classifier with ``fit``/``predict_proba``; cloned per column.
        Defaults to a depth-3 :class:`DecisionTree`.
    code :
        ``"ovr"``, ``"ovo"``, ``"exhaustive"``, ``"dense"``, ``"sparse"`` or a
        ``(n_classes, L)`` array with entries in {-1, 0, 1}.
    code_size :
        Columns per class for random codes (``L = ceil(code_size * K)``).
    decoding :
        ``"loss"`` (exponential loss on margins) or ``"hamming"``.
    n_candidates :
        Random code matrices drawn; the one with largest min distance is kept.
    """

    def __init__(
        self,
        base_estimator=None,
        code: Union[str, np.ndarray] = "dense",
        code_size: Optional[float] = None,
        decoding: str = "loss",
        n_candidates: int = 100,
        random_state: Optional[int] = None,
    ) -> None:
        self.base_estimator = base_estimator
        self.code = code
        self.code_size = code_size
        self.decoding = decoding
        self.n_candidates = n_candidates
        self.random_state = random_state

    def fit(self, X, y) -> "ECOCClassifier":
        if self.decoding not in ("loss", "hamming"):
            raise ValueError("decoding must be 'loss' or 'hamming'")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        self.classes_ = np.unique(y)
        K = self.classes_.shape[0]
        if K < 2:
            raise ValueError("ECOC needs at least two classes")
        y_idx = np.searchsorted(self.classes_, y)
        if isinstance(self.code, str):
            M = make_code_matrix(
                K, self.code, self.code_size, self.n_candidates, self.random_state
            )
        else:
            M = np.asarray(self.code, dtype=float)
            if M.shape[0] != K or M.ndim != 2:
                raise ValueError(f"code matrix must have shape ({K}, L)")
            if not np.all(np.isin(M, (-1.0, 0.0, 1.0))):
                raise ValueError("code matrix entries must be -1, 0 or 1")
            for l in range(M.shape[1]):
                if not (np.any(M[:, l] > 0) and np.any(M[:, l] < 0)):
                    raise ValueError(f"code column {l} must contain both +1 and -1")
        self.code_matrix_ = M
        self.n_features_in_ = X.shape[1]
        base = self.base_estimator if self.base_estimator is not None else DecisionTree(max_depth=3)
        self.estimators_ = []
        for l in range(M.shape[1]):
            target = M[y_idx, l]
            mask = target != 0
            est = clone_estimator(base)
            if hasattr(est, "random_state") and self.random_state is not None:
                est.random_state = self.random_state + l
            est.fit(X[mask], (target[mask] > 0).astype(int))
            self.estimators_.append(est)
        return self

    def _check(self):
        if not hasattr(self, "estimators_"):
            raise RuntimeError("ECOCClassifier is not fitted")

    def decision_margins(self, X) -> np.ndarray:
        """Per-learner margins ``f_l = 2 P(+1) - 1`` in [-1, 1], shape (n, L)."""
        self._check()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        out = np.empty((X.shape[0], len(self.estimators_)))
        for l, est in enumerate(self.estimators_):
            proba = np.asarray(est.predict_proba(X))
            classes = list(getattr(est, "classes_", [0, 1]))
            p_pos = proba[:, classes.index(1)] if 1 in classes else np.zeros(X.shape[0])
            out[:, l] = 2.0 * p_pos - 1.0
        return out

    def decode_losses(self, X) -> np.ndarray:
        """Decoding loss of every class (lower is better), shape (n, K)."""
        f = self.decision_margins(X)
        M = self.code_matrix_
        if self.decoding == "hamming":
            s = np.sign(f)
            ham = ((1.0 - s[:, None, :] * M[None]) / 2.0).sum(axis=2)
            # Break Hamming ties (common for one-vs-rest) by the soft margins;
            # the tie-break term is < 1e-3 so it never overrides a real distance gap.
            soft = (1.0 - f[:, None, :] * M[None]).sum(axis=2) / (2.0 * M.shape[1])
            return ham + 1e-3 * soft
        return np.exp(-(f[:, None, :] * M[None])).sum(axis=2)

    def predict_proba(self, X) -> np.ndarray:
        z = -self.decode_losses(X)
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        losses = self.decode_losses(X)
        return self.classes_[np.argmin(losses, axis=1)]

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))
