"""Mixture of experts regression fitted by EM (Jordan & Jacobs, 1994).

A mixture of experts splits the input space softly between ``K`` local
models. A softmax **gating network** gives each expert a mixing weight that
depends on ``x``:

    g_k(x) = softmax(V x)_k,
    p(y | x) = sum_k g_k(x) N(y | w_k^T x + b_k, sigma_k^2).

Each expert is a linear (ridge) regression with its own noise variance. The
model therefore captures piecewise-linear trends, regime changes and
input-dependent noise that a single linear model cannot.

Fitting uses Expectation-Maximisation, which is the ensemble's
"divide and conquer" learning rule (Jordan & Jacobs 1994):

* **E-step**: responsibilities
  ``h_ik = g_k(x_i) N(y_i | mu_ik, sigma_k^2) / p(y_i | x_i)``.
* **M-step (experts)**: weighted ridge regression of ``y`` on ``x`` with weights
  ``h_ik`` (penalty ``alpha * sigma_k^2``, i.e. a Gaussian prior on the slopes),
  then ``sigma_k^2 = sum_i h_ik r_ik^2 / sum_i h_ik`` (floored).
* **M-step (gate)**: soft-target multinomial logistic regression of ``h`` on
  ``x``. This uses Böhning's (1992) quadratic-bound MM step. The Hessian of the
  ridge-penalised multinomial log-likelihood is bounded by
  ``(X^T X + 2 K lambda I) ⊗ B`` with ``B = (I - 11^T / K) / 2`` on the
  ``K - 1`` free gate rows. Each step therefore needs only one fixed
  ``d x d`` solve and never decreases the objective.

Every iteration increases the penalised log-likelihood, so ``log_likelihood_``
is monotone (generalised EM). Several random restarts (``n_init``) are run, and
the one with the best final objective is kept. Initial responsibilities
come from a few k-means iterations on standardised ``[X, y]``.

Besides the point prediction ``E[y | x] = sum_k g_k mu_k``, the model gives the
full predictive mixture: ``predict_std`` (law of total variance),
``predict_gates`` and ``predict_expert_means``, plus ``score_samples`` for the
per-sample log-density.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

__all__ = ["MixtureOfExpertsRegressor"]

_LOG_2PI = float(np.log(2.0 * np.pi))


def _softmax_free(Z: np.ndarray) -> np.ndarray:
    """Softmax over ``[Z, 0]`` (last gate is the reference with zero logit)."""
    full = np.hstack([Z, np.zeros((Z.shape[0], 1))])
    full -= full.max(axis=1, keepdims=True)
    e = np.exp(full)
    return e / e.sum(axis=1, keepdims=True)


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


class MixtureOfExpertsRegressor:
    """Linear-gated mixture of linear experts, trained with EM.

    Parameters
    ----------
    n_experts : int
        Number of experts ``K`` (>= 1). ``K = 1`` reduces to ridge regression.
    max_iter : int
        Maximum EM iterations per restart.
    tol : float
        Stop when the penalised log-likelihood per sample improves by less than this.
    alpha : float
        Ridge penalty on the expert slopes (not the intercepts).
    gate_alpha : float
        Ridge penalty on the gating weights (not the intercepts).
    gate_steps : int
        Böhning MM steps for the gate in each M-step.
    min_variance : float
        Variance floor, as a fraction of ``Var(y)``.
    n_init : int
        Number of random restarts.
    random_state : int, optional
    """

    def __init__(
        self,
        n_experts: int = 3,
        max_iter: int = 200,
        tol: float = 1e-6,
        alpha: float = 1e-3,
        gate_alpha: float = 1e-3,
        gate_steps: int = 5,
        min_variance: float = 1e-4,
        n_init: int = 3,
        random_state: Optional[int] = None,
    ) -> None:
        if n_experts < 1:
            raise ValueError("n_experts must be >= 1")
        if max_iter < 1 or gate_steps < 1 or n_init < 1:
            raise ValueError("max_iter, gate_steps and n_init must be >= 1")
        if alpha < 0 or gate_alpha < 0:
            raise ValueError("alpha and gate_alpha must be non-negative")
        if min_variance <= 0:
            raise ValueError("min_variance must be positive")
        self.n_experts = n_experts
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.gate_alpha = gate_alpha
        self.gate_steps = gate_steps
        self.min_variance = min_variance
        self.n_init = n_init
        self.random_state = random_state

    # ------------------------------------------------------------------ utils
    def _design(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self.x_mean_) / self.x_scale_
        return np.hstack([Z, np.ones((Z.shape[0], 1))])

    def _log_joint(self, Phi, y, W, s2, V):
        """``log g_k(x_i) + log N(y_i | mu_ik, s2_k)`` (n x K) and the gates."""
        G = _softmax_free(Phi @ V)
        mu = Phi @ W
        log_norm = -0.5 * (_LOG_2PI + np.log(s2) + (y[:, None] - mu) ** 2 / s2)
        return np.log(np.clip(G, 1e-300, None)) + log_norm, G

    def _penalty(self, W, V) -> float:
        return 0.5 * self.alpha * float(np.sum(W[:-1] ** 2)) + 0.5 * self.gate_alpha * float(np.sum(V[:-1] ** 2))

    def _init_resp(self, Phi, y, rng) -> np.ndarray:
        K, n = self.n_experts, len(y)
        Zy = np.hstack([Phi[:, :-1], ((y - y.mean()) / (y.std() + 1e-12))[:, None]])
        centres = Zy[rng.choice(n, size=K, replace=n < K)]
        for _ in range(10):
            d = ((Zy[:, None, :] - centres[None]) ** 2).sum(axis=2)
            lab = d.argmin(axis=1)
            for k in range(K):
                if np.any(lab == k):
                    centres[k] = Zy[lab == k].mean(axis=0)
        R = np.full((n, K), 0.1 / max(K - 1, 1)) if K > 1 else np.ones((n, 1))
        if K > 1:
            R[np.arange(n), lab] = 0.9
        return R / R.sum(axis=1, keepdims=True)

    def _m_step(self, Phi, y, R, V, s2_prev, y_var):
        n, p = Phi.shape
        K = self.n_experts
        W = np.zeros((p, K))
        s2 = np.zeros(K)
        pen = self.alpha * np.eye(p)
        pen[-1, -1] = 0.0
        for k in range(K):
            h = R[:, k]
            # maximise sum_i h_ik log N(y_i | w^T phi_i, s2) - alpha/2 |w|^2 given the current s2
            A = Phi.T @ (Phi * h[:, None]) + s2_prev[k] * pen + 1e-10 * np.eye(p)
            W[:, k] = np.linalg.solve(A, Phi.T @ (h * y))
            hs = h.sum()
            r2 = (y - Phi @ W[:, k]) ** 2
            s2[k] = (h @ r2) / hs if hs > 1e-12 else y_var
        s2 = np.maximum(s2, self.min_variance * y_var)
        if K > 1:
            gpen = 2.0 * K * self.gate_alpha * np.eye(p)
            gpen[-1, -1] = 0.0
            A_inv = np.linalg.inv(Phi.T @ Phi + gpen + 1e-8 * np.eye(p))
            B = 0.5 * (np.eye(K - 1) - np.ones((K - 1, K - 1)) / K)
            B_inv = np.linalg.inv(B)
            mask = np.ones((p, 1))
            mask[-1] = 0.0
            for _ in range(self.gate_steps):
                G = _softmax_free(Phi @ V)
                grad = Phi.T @ (R[:, :-1] - G[:, :-1]) - self.gate_alpha * V * mask
                V = V + A_inv @ grad @ B_inv
        return W, s2, V

    def _fit_once(self, Phi, y, rng):
        n, p = Phi.shape
        K = self.n_experts
        y_var = float(np.var(y)) if np.var(y) > 0 else 1.0
        R = self._init_resp(Phi, y, rng)
        V = np.zeros((p, K - 1))
        s2 = np.full(K, y_var)
        history: List[float] = []
        converged = False
        for it in range(self.max_iter):
            W, s2, V = self._m_step(Phi, y, R, V, s2, y_var)
            LJ, _ = self._log_joint(Phi, y, W, s2, V)
            ll = _logsumexp(LJ, axis=1)
            obj = float(ll.sum()) - self._penalty(W, V)
            R = np.exp(LJ - ll[:, None])
            history.append(obj)
            if it > 0 and abs(history[-1] - history[-2]) < self.tol * n:
                converged = True
                break
        return W, s2, V, history, converged

    # -------------------------------------------------------------------- API
    def fit(self, X, y) -> "MixtureOfExpertsRegressor":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y, dtype=np.float64).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if X.shape[0] < 2:
            raise ValueError("need at least two samples")
        if not (np.isfinite(X).all() and np.isfinite(y).all()):
            raise ValueError("X and y must be finite")
        self.n_features_in_ = X.shape[1]
        self.x_mean_ = X.mean(axis=0)
        sc = X.std(axis=0)
        self.x_scale_ = np.where(sc > 0, sc, 1.0)
        Phi = self._design(X)
        rng = np.random.default_rng(self.random_state)
        best = None
        for _ in range(self.n_init):
            W, s2, V, hist, conv = self._fit_once(Phi, y, rng)
            if best is None or hist[-1] > best[3][-1]:
                best = (W, s2, V, hist, conv)
        self.coef_, self.variances_, self.gate_coef_, self.log_likelihood_, self.converged_ = best
        self.n_iter_ = len(self.log_likelihood_)
        return self

    def _check(self, X) -> np.ndarray:
        if not hasattr(self, "coef_"):
            raise RuntimeError("MixtureOfExpertsRegressor is not fitted")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"expected {self.n_features_in_} features, got {X.shape[1]}")
        return self._design(X)

    def predict_gates(self, X) -> np.ndarray:
        """Gating probabilities ``g_k(x)`` (n x K)."""
        return _softmax_free(self._check(X) @ self.gate_coef_)

    def predict_expert_means(self, X) -> np.ndarray:
        """Each expert's mean prediction ``mu_k(x)`` (n x K)."""
        return self._check(X) @ self.coef_

    def predict(self, X) -> np.ndarray:
        """Predictive mean ``sum_k g_k(x) mu_k(x)``."""
        return np.sum(self.predict_gates(X) * self.predict_expert_means(X), axis=1)

    def predict_std(self, X) -> np.ndarray:
        """Predictive standard deviation of the mixture (law of total variance)."""
        G, M = self.predict_gates(X), self.predict_expert_means(X)
        mean = np.sum(G * M, axis=1, keepdims=True)
        var = np.sum(G * (self.variances_[None, :] + (M - mean) ** 2), axis=1)
        return np.sqrt(var)

    def score_samples(self, X, y) -> np.ndarray:
        """Per-sample predictive log-density ``log p(y_i | x_i)``."""
        Phi = self._check(X)
        y = np.asarray(y, dtype=np.float64).ravel()
        LJ, _ = self._log_joint(Phi, y, self.coef_, self.variances_, self.gate_coef_)
        return _logsumexp(LJ, axis=1)

    def responsibilities(self, X, y) -> np.ndarray:
        """Posterior expert responsibilities ``h_ik`` given ``(x_i, y_i)``."""
        Phi = self._check(X)
        y = np.asarray(y, dtype=np.float64).ravel()
        LJ, _ = self._log_joint(Phi, y, self.coef_, self.variances_, self.gate_coef_)
        return np.exp(LJ - _logsumexp(LJ, axis=1)[:, None])

    def score(self, X, y) -> float:
        """R^2 of the predictive mean."""
        y = np.asarray(y, dtype=np.float64).ravel()
        resid = y - self.predict(X)
        ss = np.sum((y - y.mean()) ** 2)
        return float(1.0 - resid @ resid / ss) if ss > 0 else 0.0
