"""Cascade forest (gcForest-style deep forest) classifier.

Zhou & Feng (2017), "Deep Forest: Towards an Alternative to Deep Neural
Networks", build a layer-wise cascade of tree ensembles. Every layer holds
random forests and completely-random (Extra-Trees) forests. Each forest emits
a class-probability vector. Those vectors are appended to the raw features and
passed to the next layer, so later layers can correct the earlier ones.

To keep the augmented features from leaking the training labels, the class
vectors used for training come from k-fold out-of-fold predictions. The same
out-of-fold estimates give each layer a validation accuracy. The cascade keeps
adding layers until that accuracy fails to improve by more than ``tol`` for
``early_stopping_rounds`` consecutive layers. It then truncates to the best
layer, so model depth is chosen from the data instead of being fixed in
advance.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .extra_trees import ExtraTreesClassifier
from .random_forest import RandomForestClassifier
from .utils import clone_estimator

__all__ = ["CascadeForestClassifier"]


def _stratified_folds(y_idx: np.ndarray, n_folds: int, rng: np.random.Generator) -> np.ndarray:
    """Return a fold id per sample, assigning each class round-robin after shuffling."""
    fold = np.empty(y_idx.shape[0], dtype=int)
    offset = 0
    for c in np.unique(y_idx):
        members = np.flatnonzero(y_idx == c)
        rng.shuffle(members)
        fold[members] = (np.arange(members.shape[0]) + offset) % n_folds
        offset += members.shape[0]
    return fold


class CascadeForestClassifier:
    """Layer-wise cascade of random / completely-random forests (deep forest).

    Parameters
    ----------
    n_estimators :
        Trees per forest.
    n_forests :
        Number of (random forest, Extra-Trees) *pairs* per layer. Each layer
        therefore has ``2 * n_forests`` forests.
    max_depth :
        Depth limit of every tree (``None`` grows trees until the leaves are pure).
    max_layers :
        Hard cap on cascade depth.
    n_folds :
        Folds used to produce the out-of-fold class vectors and layer scores.
    early_stopping_rounds :
        Stop after this many consecutive layers without improvement.
    tol :
        Minimum out-of-fold accuracy gain that counts as an improvement.
    random_state :
        Seed for reproducibility.

    Attributes
    ----------
    classes_ : the sorted class labels.
    layers_ : the fitted layers. Each one is a list of forests trained on all of
        that layer's input.
    layer_scores_ : the out-of-fold accuracy of each grown layer.
    best_layer_ : the number of layers kept (1-based) and used for prediction.
    n_layers_ : equal to ``best_layer_``.
    """

    def __init__(
        self,
        n_estimators: int = 30,
        n_forests: int = 1,
        max_depth: Optional[int] = 8,
        max_layers: int = 10,
        n_folds: int = 3,
        early_stopping_rounds: int = 1,
        tol: float = 0.0,
        random_state: Optional[int] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.n_forests = n_forests
        self.max_depth = max_depth
        self.max_layers = max_layers
        self.n_folds = n_folds
        self.early_stopping_rounds = early_stopping_rounds
        self.tol = tol
        self.random_state = random_state
        self.classes_: Optional[np.ndarray] = None
        self.layers_: List[list] = []
        self.layer_scores_: List[float] = []
        self.best_layer_: Optional[int] = None
        self.n_layers_: Optional[int] = None

    # ------------------------------------------------------------------ utils
    def _validate(self) -> None:
        if int(self.n_estimators) < 1:
            raise ValueError("n_estimators must be >= 1")
        if int(self.n_forests) < 1:
            raise ValueError("n_forests must be >= 1")
        if int(self.max_layers) < 1:
            raise ValueError("max_layers must be >= 1")
        if int(self.n_folds) < 2:
            raise ValueError("n_folds must be >= 2")
        if int(self.early_stopping_rounds) < 1:
            raise ValueError("early_stopping_rounds must be >= 1")
        if self.tol < 0:
            raise ValueError("tol must be >= 0")

    def _make_forests(self, seed_rng: np.random.Generator) -> list:
        forests = []
        for _ in range(int(self.n_forests)):
            s1, s2 = (int(v) for v in seed_rng.integers(0, 2**31 - 1, size=2))
            forests.append(
                RandomForestClassifier(
                    n_estimators=int(self.n_estimators), max_depth=self.max_depth, random_state=s1
                )
            )
            # Completely-random forest: random thresholds on one random feature.
            forests.append(
                ExtraTreesClassifier(
                    n_estimators=int(self.n_estimators),
                    max_depth=self.max_depth,
                    max_features=1,
                    random_state=s2,
                )
            )
        return forests

    def _aligned_proba(self, forest, X: np.ndarray) -> np.ndarray:
        """Forest probabilities mapped onto the global class-index columns."""
        p = forest.predict_proba(X)
        out = np.zeros((X.shape[0], self.classes_.shape[0]))
        out[:, np.asarray(forest.classes_, dtype=int)] = p
        return out

    # -------------------------------------------------------------------- API
    def fit(self, X, y) -> "CascadeForestClassifier":
        self._validate()
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2-D")
        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y have inconsistent lengths")
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        if self.classes_.shape[0] < 2:
            raise ValueError("need at least two classes")
        if np.bincount(y_idx).min() < int(self.n_folds):
            raise ValueError("every class needs at least n_folds samples")

        rng = np.random.default_rng(self.random_state)
        n, k = X.shape[0], self.classes_.shape[0]
        self.layers_, self.layer_scores_ = [], []
        best_score, best_layer, stale = -np.inf, 0, 0
        augmented = X
        for _layer in range(int(self.max_layers)):
            forests = self._make_forests(rng)
            folds = _stratified_folds(y_idx, int(self.n_folds), rng)
            oof = np.zeros((len(forests), n, k))
            for f_i, forest in enumerate(forests):
                for fold in range(int(self.n_folds)):
                    tr, te = folds != fold, folds == fold
                    est = clone_estimator(forest)
                    est.fit(augmented[tr], y_idx[tr])
                    oof[f_i, te] = self._aligned_proba(est, augmented[te])
                forest.fit(augmented, y_idx)
            score = float(np.mean(oof.mean(axis=0).argmax(axis=1) == y_idx))
            self.layers_.append(forests)
            self.layer_scores_.append(score)
            if score > best_score + self.tol:
                best_score, best_layer, stale = score, len(self.layers_), 0
            else:
                stale += 1
                if stale >= int(self.early_stopping_rounds):
                    break
            class_vectors = oof.transpose(1, 0, 2).reshape(n, -1)
            augmented = np.hstack([X, class_vectors])

        self.layers_ = self.layers_[:best_layer]
        self.best_layer_ = best_layer
        self.n_layers_ = best_layer
        return self

    def _check_fitted(self) -> None:
        if self.best_layer_ is None:
            raise RuntimeError("CascadeForestClassifier must be fitted before predict")

    def layer_proba(self, X) -> List[np.ndarray]:
        """The averaged class probabilities produced by each kept layer, in order."""
        self._check_fitted()
        X = np.asarray(X, dtype=float)
        augmented = X
        out = []
        for forests in self.layers_:
            probs = np.stack([self._aligned_proba(f, augmented) for f in forests])
            out.append(probs.mean(axis=0))
            augmented = np.hstack([X, probs.transpose(1, 0, 2).reshape(X.shape[0], -1)])
        return out

    def predict_proba(self, X) -> np.ndarray:
        return self.layer_proba(X)[-1]

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def score(self, X, y) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))

