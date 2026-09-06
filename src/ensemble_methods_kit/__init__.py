"""ensemble-methods-kit: a from-scratch NumPy toolkit of ensemble learning methods.

The package re-implements common ensemble strategies (bagging, random forest,
gradient boosting, voting, stacking and blending) on top of a small CART
decision tree so that the whole workflow is dependency-light and easy to follow.
"""

from .utils import (
    DecisionTree,
    accuracy_score,
    log_loss,
    mean_squared_error,
    r2_score,
    train_test_split,
)
from .bagging import BaggingClassifier
from .random_forest import RandomForestClassifier

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DecisionTree",
    "BaggingClassifier",
    "RandomForestClassifier",
    "accuracy_score",
    "log_loss",
    "mean_squared_error",
    "r2_score",
    "train_test_split",
]
