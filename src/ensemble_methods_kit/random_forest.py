"""Random forest classifier built on the kit's CART decision tree."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from .bagging import BaggingClassifier
from .utils import DecisionTree

__all__ = ["RandomForestClassifier"]


class RandomForestClassifier(BaggingClassifier):
    """A random forest of decision trees.

    Like :class:`BaggingClassifier` but every tree considers a random subset
    of features at each split, which decorrelates the trees and reduces the
    variance of the averaged prediction.

    Parameters
    ----------
    n_estimators :
        Number of trees.
    max_depth, min_samples_split, min_impurity_decrease :
        Forwarded to each decision tree.
    max_features :
        Feature sub-sampling strategy passed to each tree.  ``"sqrt"`` (the
        default) uses :math:`\\sqrt{n_features}` features; ``"log2"`` uses
        :math:`\\log_2 n_features`; an int is the exact count.
    max_samples :
        Fraction of the training set sampled per tree.
    bootstrap :
        Whether to bootstrap.
    random_state :
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 5,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.0,
        max_features: Union[int, str] = "sqrt",
        max_samples: float = 1.0,
        bootstrap: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        base = DecisionTree(
            criterion="gini",
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_impurity_decrease=min_impurity_decrease,
            max_features=max_features,
        )
        super().__init__(
            base_estimator=base,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=1.0,
            bootstrap=bootstrap,
            random_state=random_state,
        )
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.max_samples = max_samples
        self.bootstrap = bootstrap

    def _clone_tree(self, seed: int) -> DecisionTree:
        return DecisionTree(
            criterion="gini",
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=self.max_features,
            random_state=seed,
        )
