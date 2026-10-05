"""Rotation Forest ensemble (Rodriguez, Kuncheva & Alonso, 2006).

Each base tree is trained in a rotated feature space built by:
1. randomly partitioning the ``p`` features into ``n_groups`` disjoint subsets,
2. drawing a bootstrap of training rows (optionally restricted to a random
   subset of classes) for each subset,
3. running PCA on that subset and scattering the loadings back into a
   sparse ``p × p`` rotation matrix ``R``,
4. fitting a CART tree on ``X @ R``.

Predictions average soft class probabilities across trees (same aggregation
as :class:`~ensemble_methods_kit.bagging.BaggingClassifier`).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .utils import DecisionTree, mean_decrease_impurity

__all__ = ["RotationForestClassifier"]


def _pca_loadings(X: np.ndarray) -> np.ndarray:
    """Return PCA loading matrix ``(n_features, n_components)`` for ``X``.

    Components with near-zero eigenvalues are dropped so degenerate subsets
    (constant columns after bootstrap) still produce a usable basis. When
    every eigenvalue is zero the identity of matching width is returned.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError("PCA input must be a 2-d array with ≥1 feature")
    n, p = X.shape
    if n == 0:
        return np.eye(p, dtype=np.float64)
    xc = X - X.mean(axis=0, keepdims=True)
    # Economy SVD on centred data; loadings are V.
    # For p >> n use Xc @ Xc.T path via SVD of Xc.
    try:
        _, s, vt = np.linalg.svd(xc, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.eye(p, dtype=np.float64)
    keep = s > 1e-10
    if not np.any(keep):
        return np.eye(p, dtype=np.float64)
    return vt[keep].T  # (p, k)


class RotationForestClassifier:
    """Rotation Forest classifier (Rodriguez et al., 2006).

    Parameters
    ----------
    base_estimator :
        A :class:`DecisionTree` template. ``None`` defaults to a deep gini
        tree that sees every rotated coordinate.
    n_estimators :
        Number of rotated trees.
    n_features_per_subset :
        Target size of each random feature group. The last group may be
        smaller when ``n_features`` is not divisible by this value.
        Ignored when ``n_subsets`` is set.
    n_subsets :
        Optional explicit number of feature groups. When given, features
        are split as evenly as possible into this many groups.
    max_samples :
        Fraction of training rows drawn (with replacement) for each PCA
        subset. Classic Rotation Forest uses ``0.75``.
    bootstrap_classes :
        When ``True``, each PCA subset is built from a random 50–100% of
        the class labels (Rodriguez et al.); when ``False`` every class is
        kept.
    random_state :
        Seed for reproducible rotations / bootstraps.

    Attributes
    ----------
    estimators_ :
        Fitted decision trees.
    rotation_matrices_ :
        Per-tree rotation matrices of shape ``(n_features, n_features)``.
    classes_ / n_classes_ :
        Unique training labels and their count.
    """

    def __init__(
        self,
        base_estimator: Optional[DecisionTree] = None,
        n_estimators: int = 10,
        n_features_per_subset: int = 3,
        n_subsets: Optional[int] = None,
        max_samples: float = 0.75,
        bootstrap_classes: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        if n_features_per_subset < 1:
            raise ValueError("n_features_per_subset must be at least 1")
        if n_subsets is not None and int(n_subsets) < 1:
            raise ValueError("n_subsets must be at least 1")
        if not 0.0 < float(max_samples) <= 1.0:
            raise ValueError("max_samples must be in (0, 1]")
        self.base_estimator = base_estimator
        self.n_estimators = int(n_estimators)
        self.n_features_per_subset = int(n_features_per_subset)
        self.n_subsets = None if n_subsets is None else int(n_subsets)
        self.max_samples = float(max_samples)
        self.bootstrap_classes = bool(bootstrap_classes)
        self.random_state = random_state
        self.estimators_: list = []
        self.rotation_matrices_: List[np.ndarray] = []
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: Optional[int] = None
        self._n_features_in_: Optional[int] = None

    def _clone_tree(self, seed: int) -> DecisionTree:
        base = self.base_estimator
        common = dict(
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_impurity_decrease=0.0,
            max_features=None,
            splitter="best",
            random_state=seed,
        )
        if isinstance(base, DecisionTree):
            common = dict(
                criterion=base.criterion,
                max_depth=base.max_depth,
                min_samples_split=base.min_samples_split,
                min_impurity_decrease=base.min_impurity_decrease,
                max_features=None,
                splitter=getattr(base, "splitter", "best"),
                random_state=seed,
            )
        return DecisionTree(**common)

    def _feature_groups(self, n_features: int, rng: np.random.Generator) -> List[np.ndarray]:
        order = rng.permutation(n_features)
        if self.n_subsets is not None:
            k = min(self.n_subsets, n_features)
            return [np.sort(g) for g in np.array_split(order, k) if len(g) > 0]
        size = self.n_features_per_subset
        groups = [np.sort(order[i : i + size]) for i in range(0, n_features, size)]
        return [g for g in groups if len(g) > 0]

    def _row_indices_for_pca(
        self, y: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        classes = np.unique(y)
        if self.bootstrap_classes and classes.size > 1:
            # Draw between 50% and 100% of classes (at least 1).
            n_keep = max(1, int(np.ceil(rng.uniform(0.5, 1.0) * classes.size)))
            n_keep = min(n_keep, classes.size)
            keep = rng.choice(classes, size=n_keep, replace=False)
            mask = np.isin(y, keep)
            pool = np.flatnonzero(mask)
        else:
            pool = np.arange(y.shape[0])
        if pool.size == 0:
            pool = np.arange(y.shape[0])
        n_draw = max(1, int(round(self.max_samples * pool.size)))
        n_draw = min(n_draw, pool.size) if pool.size < n_draw else n_draw
        # With replacement as in the original paper's bootstrap of instances.
        return pool[rng.integers(0, pool.size, size=n_draw)]

    def _build_rotation(
        self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        n_features = X.shape[1]
        R = np.zeros((n_features, n_features), dtype=np.float64)
        col = 0
        for feat_idx in self._feature_groups(n_features, rng):
            rows = self._row_indices_for_pca(y, rng)
            Xi = X[np.ix_(rows, feat_idx)]
            # Drop near-constant columns inside the subset for numerical PCA.
            std = Xi.std(axis=0)
            useful = std > 1e-12
            if not np.any(useful):
                loadings = np.eye(feat_idx.size, dtype=np.float64)
            else:
                Xi_u = Xi[:, useful]
                L = _pca_loadings(Xi_u)  # (k_useful, k_comp)
                loadings = np.zeros((feat_idx.size, L.shape[1]), dtype=np.float64)
                loadings[useful] = L
                # Pad with unused original axes so the block stays full rank-ish.
                if loadings.shape[1] < feat_idx.size:
                    pad = np.eye(feat_idx.size, dtype=np.float64)[:, loadings.shape[1] :]
                    # Orthogonalise pad against existing loadings via QR-ish projection.
                    for j in range(pad.shape[1]):
                        v = pad[:, j]
                        for c in range(loadings.shape[1]):
                            v = v - np.dot(v, loadings[:, c]) * loadings[:, c]
                        nrm = np.linalg.norm(v)
                        if nrm > 1e-12:
                            pad[:, j] = v / nrm
                        else:
                            pad[:, j] = 0.0
                    if np.any(np.abs(pad) > 1e-12):
                        loadings = np.hstack([loadings, pad[:, np.any(np.abs(pad) > 1e-12, axis=0)]])
            # Place loadings into successive columns of R.
            n_comp = min(loadings.shape[1], n_features - col)
            for j in range(n_comp):
                R[feat_idx, col + j] = loadings[:, j]
            col += n_comp
            if col >= n_features:
                break
        # Fill any unused columns with leftover identity directions.
        if col < n_features:
            used_rows = np.where(np.abs(R[:, :col]).sum(axis=1) > 1e-12)[0]
            free = [i for i in range(n_features) if i not in set(used_rows.tolist())]
            for i in free:
                if col >= n_features:
                    break
                R[i, col] = 1.0
                col += 1
            while col < n_features:
                R[col % n_features, col] = 1.0
                col += 1
        return R

    def fit(self, X, y) -> "RotationForestClassifier":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample")
        self.classes_ = np.unique(y)
        self.n_classes_ = int(self.classes_.shape[0])
        self._n_features_in_ = int(X.shape[1])
        self.estimators_ = []
        self.rotation_matrices_ = []
        rng = np.random.default_rng(self.random_state)
        for _ in range(self.n_estimators):
            R = self._build_rotation(X, y, rng)
            Xr = X @ R
            tree = self._clone_tree(seed=int(rng.integers(0, 2**31 - 1)))
            tree.fit(Xr, y)
            self.estimators_.append(tree)
            self.rotation_matrices_.append(R)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if not self.estimators_:
            raise RuntimeError("Estimator is not fitted yet")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self._n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self._n_features_in_}"
            )
        probs = [
            tree.predict_proba(X @ R)
            for tree, R in zip(self.estimators_, self.rotation_matrices_)
        ]
        return np.mean(probs, axis=0)

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]

    @property
    def feature_importances_(self) -> np.ndarray:
        """Approximate MDI importances projected back to the original space.

        Each tree's importances live in the rotated coordinates; they are
        mapped with ``|R| @ importances_rotated`` and averaged.
        """
        if not self.estimators_ or self._n_features_in_ is None:
            raise RuntimeError("Estimator is not fitted yet")
        n_features = self._n_features_in_
        acc = np.zeros(n_features, dtype=np.float64)
        for tree, R in zip(self.estimators_, self.rotation_matrices_):
            local = mean_decrease_impurity([tree])
            # local length = tree.n_features_in_ (== n_features after rotation)
            if local.shape[0] != n_features:
                continue
            acc += np.abs(R) @ local
        total = float(acc.sum())
        if total <= 0.0:
            return np.full(n_features, 1.0 / n_features, dtype=np.float64)
        return acc / total
