"""Super Learner: cross-validated optimal convex combination of learners.

van der Laan, Polley & Hubbard (2007) show that choosing ensemble weights
by minimising the *cross-validated* risk of a weighted combination of
candidate learners performs asymptotically as well as the best possible
weighted combination (the "oracle"). The recipe:

1. Split the data into ``V`` folds. For every candidate learner, fit on
   ``V - 1`` folds and predict the held-out fold, which gives an ``n x L``
   matrix ``Z`` of out-of-fold predictions (``n x L x K`` probabilities for
   classifiers).
2. Choose weights ``alpha`` by minimising the loss of ``Z alpha`` against
   ``y``:

   * ``"nnls"``: non-negative least squares, rescaled to sum to one (the
     default ``method.NNLS`` of the R ``SuperLearner`` package);
   * ``"convex"``: least squares constrained to the simplex
     (``method.CC_LS``);
   * ``"log_loss"`` (classifiers): negative log-likelihood of the mixed
     probabilities on the simplex (``method.CC_nloglik``);
   * ``"discrete"``: all weight on the learner with the lowest CV risk
     (the "discrete super learner", i.e. CV model selection).
3. Refit every learner with non-zero weight on all the data. A prediction is
   the weighted combination of the refitted learners.

Everything is NumPy, including Lawson-Hanson NNLS and the simplex solvers.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .stacking import _kfold_indices
from .utils import clone_estimator

__all__ = [
    "SuperLearnerClassifier",
    "SuperLearnerRegressor",
    "nnls",
    "project_to_simplex",
]


def nnls(A: np.ndarray, b: np.ndarray, max_iter: Optional[int] = None, tol: float = 1e-10) -> np.ndarray:
    """Lawson-Hanson active-set solution of ``min ||A x - b||_2, x >= 0``."""
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).ravel()
    m, n = A.shape
    if b.shape[0] != m:
        raise ValueError("A and b have incompatible shapes")
    max_iter = 3 * n if max_iter is None else max_iter
    x = np.zeros(n)
    passive = np.zeros(n, dtype=bool)
    w = A.T @ (b - A @ x)
    it = 0
    while (~passive).any() and np.max(w[~passive]) > tol:
        j = np.flatnonzero(~passive)[np.argmax(w[~passive])]
        passive[j] = True
        while True:
            it += 1
            if it > max_iter * 10:
                return x
            s = np.zeros(n)
            s[passive] = np.linalg.lstsq(A[:, passive], b, rcond=None)[0]
            if np.all(s[passive] > tol):
                x = s
                break
            bad = passive & (s <= tol)
            alpha = np.min(x[bad] / (x[bad] - s[bad]))
            x = x + alpha * (s - x)
            passive &= x > tol
            x[~passive] = 0.0
        w = A.T @ (b - A @ x)
    return x


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto ``{x >= 0, sum x = 1}`` (Duchi et al., 2008)."""
    v = np.asarray(v, dtype=np.float64)
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    ind = np.arange(1, v.size + 1)
    rho = np.flatnonzero(u - css / ind > 0)[-1]
    theta = css[rho] / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def _convex_least_squares(Z: np.ndarray, y: np.ndarray, n_iter: int = 5000, tol: float = 1e-12) -> np.ndarray:
    """Accelerated projected gradient for ``min ||Z a - y||^2`` on the simplex."""
    L = Z.shape[1]
    G = Z.T @ Z
    c = Z.T @ y
    step = 1.0 / max(np.linalg.eigvalsh(G).max(), 1e-12)
    a = np.full(L, 1.0 / L)
    z = a.copy()
    t = 1.0
    for _ in range(n_iter):
        a_new = project_to_simplex(z - step * (G @ z - c))
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = a_new + ((t - 1.0) / t_new) * (a_new - a)
        if np.max(np.abs(a_new - a)) < tol:
            a = a_new
            break
        a, t = a_new, t_new
    return a


def _convex_log_loss(P: np.ndarray, y_int: np.ndarray, n_iter: int = 3000, tol: float = 1e-12) -> np.ndarray:
    """Exponentiated-gradient minimisation of ``-mean log sum_l a_l P[i, l, y_i]``."""
    n, L, _ = P.shape
    Py = np.clip(P[np.arange(n), :, y_int], 1e-15, 1.0)  # (n, L)
    a = np.full(L, 1.0 / L)
    eta = 1.0
    prev = np.inf
    for _ in range(n_iter):
        mix = Py @ a
        loss = -np.mean(np.log(mix))
        grad = -(Py / mix[:, None]).mean(axis=0)
        a_new = a * np.exp(-eta * (grad - grad.min()))
        a_new /= a_new.sum()
        new_loss = -np.mean(np.log(Py @ a_new))
        if new_loss > loss + 1e-15:
            eta *= 0.5
            continue
        a = a_new
        if prev - new_loss < tol:
            break
        prev = new_loss
    return a


def _normalise(w: np.ndarray, risks: np.ndarray) -> np.ndarray:
    s = w.sum()
    if s <= 0:  # NNLS returned all zeros: fall back to the discrete super learner
        w = np.zeros_like(w)
        w[int(np.argmin(risks))] = 1.0
        return w
    return w / s


class _SuperLearnerBase:
    def __init__(
        self,
        estimators: Sequence[Tuple[str, object]],
        method: str = "nnls",
        cv: int = 10,
        random_state: Optional[int] = None,
        trim: float = 0.0,
    ) -> None:
        self.estimators = list(estimators)
        self.method = method
        self.cv = cv
        self.random_state = random_state
        self.trim = trim

    def _validate(self, X, y):
        if not self.estimators:
            raise ValueError("estimators must be a non-empty list of (name, estimator)")
        names = [n for n, _ in self.estimators]
        if len(set(names)) != len(names):
            raise ValueError("estimator names must be unique")
        if self.method not in self._methods:
            raise ValueError(f"method must be one of {self._methods}")
        if int(self.cv) < 2:
            raise ValueError("cv must be at least 2")
        if not (0.0 <= self.trim < 1.0):
            raise ValueError("trim must be in [0, 1)")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if X.shape[0] < self.cv:
            raise ValueError("need at least cv samples")
        return X, y

    def _finish(self, X, y, Z, risks):
        w = self._weights(Z, y, risks)
        if self.trim > 0:  # drop negligible learners, renormalise
            w = np.where(w >= self.trim, w, 0.0)
            w = _normalise(w, risks)
        self.weights_ = w
        self.cv_risk_ = risks
        self.cv_predictions_ = Z
        self.names_ = [n for n, _ in self.estimators]
        self.fitted_ = {}
        for (name, est), wi in zip(self.estimators, w):
            if wi > 0:
                full = clone_estimator(est)
                full.fit(X, y)
                self.fitted_[name] = full
        self.super_learner_cv_risk_ = self._risk_of_combination(Z, y, w)
        return self

    def summary(self) -> List[Dict[str, float]]:
        """Per-learner CV risk and weight (sorted by weight, then risk)."""
        rows = [
            {"name": n, "cv_risk": float(r), "weight": float(w)}
            for n, r, w in zip(self.names_, self.cv_risk_, self.weights_)
        ]
        return sorted(rows, key=lambda r: (-r["weight"], r["cv_risk"]))


class SuperLearnerRegressor(_SuperLearnerBase):
    """Super Learner for regression (squared-error loss).

    Parameters
    ----------
    estimators :
        ``(name, estimator)`` candidates; each needs ``fit`` / ``predict``.
    method :
        ``"nnls"`` (default), ``"convex"`` or ``"discrete"``.
    cv :
        Number of folds ``V`` for the out-of-fold prediction matrix.
    trim :
        Weights below this are zeroed and the rest renormalised.

    Attributes
    ----------
    weights_, cv_risk_ (per-learner CV MSE), cv_predictions_ (n x L),
    super_learner_cv_risk_ (MSE of the weighted CV predictions; optimistic
    because the weights were fitted on them), fitted_.
    """

    _methods = ("nnls", "convex", "discrete")

    def fit(self, X, y) -> "SuperLearnerRegressor":
        X, y = self._validate(X, y)
        y = y.astype(np.float64)
        n, L = X.shape[0], len(self.estimators)
        Z = np.zeros((n, L))
        rng = np.random.default_rng(self.random_state)
        splits = _kfold_indices(n, int(self.cv), rng)
        for li, (_, est) in enumerate(self.estimators):
            for tr, va in splits:
                m = clone_estimator(est)
                m.fit(X[tr], y[tr])
                Z[va, li] = np.asarray(m.predict(X[va]), dtype=np.float64)
        risks = ((Z - y[:, None]) ** 2).mean(axis=0)
        return self._finish(X, y, Z, risks)

    def _weights(self, Z, y, risks):
        if self.method == "discrete":
            w = np.zeros(Z.shape[1])
            w[int(np.argmin(risks))] = 1.0
            return w
        if self.method == "nnls":
            return _normalise(nnls(Z, y), risks)
        return _convex_least_squares(Z, y)

    @staticmethod
    def _risk_of_combination(Z, y, w):
        return float(((Z @ w - y) ** 2).mean())

    def predict(self, X) -> np.ndarray:
        if not hasattr(self, "weights_"):
            raise RuntimeError("call fit() first")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        out = np.zeros(X.shape[0])
        for name, w in zip(self.names_, self.weights_):
            if w > 0:
                out += w * np.asarray(self.fitted_[name].predict(X), dtype=np.float64)
        return out


class SuperLearnerClassifier(_SuperLearnerBase):
    """Super Learner for classification over predicted probabilities.

    ``method``: ``"log_loss"`` (default; convex weights minimising CV
    negative log-likelihood), ``"nnls"`` (Brier / squared error on the
    one-hot labels, rescaled), ``"convex"`` (Brier on the simplex) or
    ``"discrete"`` (lowest CV log loss). Each candidate needs
    ``predict_proba``. ``cv_risk_`` holds per-learner CV log loss.
    """

    _methods = ("log_loss", "nnls", "convex", "discrete")

    def __init__(self, estimators, method: str = "log_loss", cv: int = 10, random_state=None, trim: float = 0.0):
        super().__init__(estimators, method, cv, random_state, trim)

    def fit(self, X, y) -> "SuperLearnerClassifier":
        X, y = self._validate(X, y)
        self.classes_ = np.unique(y)
        K = self.classes_.size
        if K < 2:
            raise ValueError("need at least two classes")
        y_int = np.searchsorted(self.classes_, y)
        n, L = X.shape[0], len(self.estimators)
        P = np.zeros((n, L, K))
        rng = np.random.default_rng(self.random_state)
        splits = _kfold_indices(n, int(self.cv), rng)
        for li, (_, est) in enumerate(self.estimators):
            for tr, va in splits:
                m = clone_estimator(est)
                m.fit(X[tr], y[tr])
                proba = np.asarray(m.predict_proba(X[va]), dtype=np.float64)
                # align columns when a fold misses a class
                fold_classes = np.asarray(getattr(m, "classes_", self.classes_))
                cols = np.searchsorted(self.classes_, fold_classes)
                P[va[:, None], li, cols[None, :]] = proba
        self._y_int = y_int
        Py = np.clip(P[np.arange(n), :, y_int], 1e-15, 1.0)
        risks = -np.log(Py).mean(axis=0)
        return self._finish(X, y, P, risks)

    def _weights(self, P, y, risks):
        n, L, K = P.shape
        y_int = self._y_int
        if self.method == "discrete":
            w = np.zeros(L)
            w[int(np.argmin(risks))] = 1.0
            return w
        if self.method == "log_loss":
            return _convex_log_loss(P, y_int)
        onehot = np.eye(K)[y_int]
        Zs = P.transpose(0, 2, 1).reshape(n * K, L)
        ys = onehot.reshape(n * K)
        if self.method == "nnls":
            return _normalise(nnls(Zs, ys), risks)
        return _convex_least_squares(Zs, ys)

    def _risk_of_combination(self, P, y, w):
        mix = np.clip(np.einsum("nlk,l->nk", P, w)[np.arange(P.shape[0]), self._y_int], 1e-15, 1.0)
        return float(-np.log(mix).mean())

    def predict_proba(self, X) -> np.ndarray:
        if not hasattr(self, "weights_"):
            raise RuntimeError("call fit() first")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        out = np.zeros((X.shape[0], self.classes_.size))
        for name, w in zip(self.names_, self.weights_):
            if w > 0:
                m = self.fitted_[name]
                proba = np.asarray(m.predict_proba(X), dtype=np.float64)
                cols = np.searchsorted(self.classes_, np.asarray(getattr(m, "classes_", self.classes_)))
                out[:, cols] += w * proba
        return out

    def predict(self, X) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
