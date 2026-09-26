"""Extremely randomized trees classifier built on the kit's CART tree."""

from __future__ import annotations

from typing import Optional, Union

from .bagging import BaggingClassifier, BaggingRegressor
from .utils import DecisionTree

__all__ = ["ExtraTreesClassifier", "ExtraTreesRegressor"]


class ExtraTreesClassifier(BaggingClassifier):
    """An Extra-Trees ensemble of decision trees.

    Like :class:`RandomForestClassifier` this draws a random subset of
    features at each split, but the split threshold itself is also random:
    each candidate feature gets one cut drawn uniformly from its observed
    range, and the strongest of those random cuts is kept.  Combined with
    the default of fitting every tree on the full sample (``bootstrap=False``)
    this is the Extra-Trees algorithm of Geurts, Ernst and Wehenkel (2006).

    After :meth:`fit`, :attr:`feature_importances_` is the Mean Decrease
    Impurity (MDI) ranking of the input features, averaged over the trees.

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
        Fraction of the training set sampled per tree.  Only used when
        ``bootstrap=True``.
    bootstrap :
        Whether to bootstrap.  Extra-Trees traditionally uses the full
        learning sample (``False``).
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
        bootstrap: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        base = DecisionTree(
            criterion="gini",
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_impurity_decrease=min_impurity_decrease,
            max_features=max_features,
            splitter="random",
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
            splitter="random",
            random_state=seed,
        )


class ExtraTreesRegressor(BaggingRegressor):
    """An Extra-Trees ensemble of regression trees.

    Like :class:`RandomForestRegressor` this draws a random subset of
    features at each split, but the split threshold itself is also random:
    each candidate feature gets one cut drawn uniformly from its observed
    range, and the strongest of those random cuts is kept.  Combined with
    the default of fitting every tree on the full sample (``bootstrap=False``)
    this is the Extra-Trees algorithm of Geurts, Ernst and Wehenkel (2006)
    applied to squared-error CART trees.

    After :meth:`fit`, :attr:`feature_importances_` is the Mean Decrease
    Impurity (MDI) ranking of the input features, averaged over the trees.
    Set ``oob_score=True`` to also record the out-of-bag R² (requires
    ``bootstrap=True``).

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
        Fraction of the training set sampled per tree.  Only used when
        ``bootstrap=True``.
    bootstrap :
        Whether to bootstrap.  Extra-Trees traditionally uses the full
        learning sample (``False``).
    oob_score :
        When ``True``, compute the out-of-bag R² after fitting.  Requires
        ``bootstrap=True``.
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
        bootstrap: bool = False,
        oob_score: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        base = DecisionTree(
            criterion="variance",
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_impurity_decrease=min_impurity_decrease,
            max_features=max_features,
            splitter="random",
        )
        super().__init__(
            base_estimator=base,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=1.0,
            bootstrap=bootstrap,
            oob_score=oob_score,
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
            criterion="variance",
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_impurity_decrease=self.min_impurity_decrease,
            max_features=self.max_features,
            splitter="random",
            random_state=seed,
        )

